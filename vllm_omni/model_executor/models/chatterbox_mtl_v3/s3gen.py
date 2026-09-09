# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox Multilingual V3 acoustic stage: codec ids -> 24 kHz waveform.

Wraps the vendored ``S3Token2Wav`` (conditional-flow-matching token-to-mel plus
the HiFT vocoder) with the three things serving requires and the reference
wrapper does not provide:

* **per-row reference conditioning.** Upstream batching broadcasts
  ``references[0]`` across the batch, which serves one caller's voice to
  another. Here a batch is only formed from rows that genuinely share a
  reference, and the grouping key is explicit.
* **request-scoped stochastic inputs.** ``torch.manual_seed`` in a concurrent
  server is unsafe, and the flow's ``noised_mels`` argument only replaces part
  of its noise. Each request gets its own ``torch.Generator`` for the flow noise
  and the vocoder phase/noise.
* **explicit finalization.** The initial fade, the final ``max(1, N-1)*960``
  crop and the watermark are reference behaviour and are applied here, once.

Nothing about the acoustic architecture is changed: the solver, its schedule,
the acoustic CFG rate and the vocoder are the vendored ones.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K
from vllm_omni.model_executor.models.chatterbox_mtl_v3.conditioning import ReferenceConditioning
from vllm_omni.model_executor.models.chatterbox_mtl_v3.streaming import (
    final_sample_limit,
    is_valid_codec_id,
)

logger = init_logger(__name__)


_TF32_DISABLED = False


def _disable_conv_tf32() -> None:
    """Turn off TF32 in cuDNN convolutions for the acoustic stage.

    Measured on an RTX 5090 (Blackwell, torch 2.13+cu129): with cuDNN TF32 on,
    decoding the SAME request alone versus batched with two others changed its
    waveform by 8.3e-3 against a 0.35 peak -- about 2.4%, far above float noise,
    because TF32 convolutions pick different algorithms per batch shape and the
    error is then amplified through 10 ODE solver steps and the vocoder. With
    TF32 off the same comparison drops to ~5e-6 in the mel.

    A caller's audio must not depend on who else happened to be in their batch,
    so this is a correctness setting, not a tuning knob. The AR stage is
    unaffected (it has no convolutions).
    """
    global _TF32_DISABLED
    if _TF32_DISABLED or not torch.cuda.is_available():
        return
    torch.backends.cudnn.allow_tf32 = False
    _TF32_DISABLED = True
    logger.info(
        "Chatterbox MTL V3 S3Gen: disabled cuDNN TF32 so acoustic output is independent "
        "of batch composition"
    )


class AcousticError(RuntimeError):
    """The acoustic stage refused to produce audio."""


@dataclass(frozen=True)
class AcousticRequest:
    """One acoustic decode job."""

    request_id: str
    codes: torch.Tensor                # 1-D int64, already validated to 0..6560
    conditioning: ReferenceConditioning
    seed: int
    finalize: bool = True
    #: Codec tokens already decoded for this request (incremental mode only).
    token_offset: int = 0
    #: Incremental streaming decode. Switches the flow noise to a fixed
    #: per-request bank so a prefix decoded twice at two lengths gets the same
    #: noise, and makes the row emit only samples past ``token_offset``.
    streaming: bool = False

    def batch_key(self, ragged: bool = False) -> tuple:
        """What must match for two rows to share one acoustic forward pass.

        **Default (``ragged=False``): same effective length, any voice.**
        Different reference voices batch together safely -- each row carries
        its own prompt tokens, prompt mel, speaker embedding and noise, and a
        gate asserts each row is identical to its solo decode. What may *not*
        differ is the length, because padding is what this checkpoint's token
        encoder does not isolate (see below).

        ``ragged=True`` additionally packs different lengths. **It is measurably
        unsafe on this checkpoint** and is opt-in only, for experimentation:

            batching a 40-code row with a 97-code row changed the shorter row's
            token-encoder output by 0.46 against a scale of 3.6 (13%), and its
            waveform by 40% of peak -- even though its own attention mask was
            correct (mask sums matched the solo run exactly).

        That is the "unrestricted attention over the supplied sequence" the
        port plan predicts for this encoder. Making padding safe would mean
        changing the acoustic architecture's masking, which is a model-quality
        change requiring its own qualification -- not a batching optimisation.
        """
        key = [
            self.conditioning.checkpoint_profile,
            bool(self.finalize),
            bool(self.streaming),
            int(self.token_offset),
        ]
        if not ragged:
            key.append(int(self.codes.shape[-1]))
        return tuple(key)


@dataclass
class AcousticResult:
    request_id: str
    audio: torch.Tensor          # 1-D float32, 24 kHz
    n_valid_codes: int
    watermarked: bool


def request_generator(device: torch.device, seed: int) -> torch.Generator:
    """A generator private to one request.

    Never ``torch.manual_seed``: that mutates process-global state, so two
    concurrent requests would reseed each other mid-synthesis.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
    return gen


def derive_seed(request_id: str, base_seed: int | None, purpose: str) -> int:
    """Stable per-purpose acoustic seed.

    When the caller supplies a ``seed``, the stream is derived from THAT seed
    plus the purpose and nothing else, so the same request repeated gives the
    same audio. Folding the request id in would make every call unique and
    quietly break reproducibility, which is what ``seed`` is for.

    Without a caller seed the request id is used, so concurrent requests still
    get independent noise instead of all sharing one stream.

    The flow and vocoder get separate streams from the same identity so neither
    can accidentally consume the other's draws.
    """
    h = hashlib.sha256()
    if base_seed is None:
        h.update(b"req\x00")
        h.update(request_id.encode("utf-8"))
    else:
        h.update(b"seed\x00")
        h.update(str(int(base_seed)).encode("utf-8"))
    h.update(b"\x00")
    h.update(purpose.encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big")


class ChatterboxS3Gen(torch.nn.Module):
    """The acoustic model plus its serving policy."""

    def __init__(
        self,
        *,
        checkpoint_profile: str = K.DEFAULT_CHECKPOINT_PROFILE,
        cfm_timesteps: int = K.DEFAULT_CFM_TIMESTEPS,
        apply_watermark: bool = True,
        deterministic_convolutions: bool = True,
        ragged_batching: bool = False,
        max_batch_rows: int = 8,
    ) -> None:
        super().__init__()
        if deterministic_convolutions:
            _disable_conv_tf32()
        from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.s3gen import S3Gen

        if checkpoint_profile not in K.CHECKPOINT_PROFILES:
            raise AcousticError(f"unknown checkpoint profile {checkpoint_profile!r}")
        self.checkpoint_profile = checkpoint_profile
        self.cfm_timesteps = int(cfm_timesteps)
        self.apply_watermark = bool(apply_watermark)
        self.ragged_batching = bool(ragged_batching)
        self.max_batch_rows = max(1, int(max_batch_rows))
        self.s3gen = S3Gen()
        self._watermarker = None
        #: Held-back tail per streaming request. Streaming can never revise what
        #: it already sent, so the overlap is retained here and blended with the
        #: next chunk's decode of the same region before being emitted.
        self._stream_state: dict[str, dict] = {}
        self.stream_crossfade_samples = int(K.ACOUSTIC_STREAM_CROSSFADE_SAMPLES)

    # -- weights ------------------------------------------------------------
    def load_weights_from_dir(self, model_dir: str) -> None:
        filename = K.CHECKPOINT_PROFILES[self.checkpoint_profile]["s3gen"]
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            raise AcousticError(f"acoustic checkpoint missing: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.s3gen.load_state_dict(state, strict=False)
        allowed = set(getattr(self.s3gen, "ignore_state_dict_missing", ()))
        unexplained = [m for m in missing if m not in allowed]
        if unexplained or unexpected:
            raise AcousticError(
                f"S3Gen state-dict mismatch for profile {self.checkpoint_profile!r} "
                f"({filename}): missing={unexplained[:8]} unexpected={list(unexpected)[:8]}"
            )
        logger.info(
            "Chatterbox MTL V3 S3Gen: loaded %s (profile %s), %d tensors",
            filename, self.checkpoint_profile, len(state),
        )

    # -- watermark ----------------------------------------------------------
    def _get_watermarker(self):
        if self._watermarker is None:
            import perth

            self._watermarker = perth.PerthImplicitWatermarker()
        return self._watermarker

    def watermark(self, audio: np.ndarray) -> np.ndarray:
        """Reference Perth watermark. Never silently skipped to look faster."""
        if not self.apply_watermark:
            return audio
        return self._get_watermarker().apply_watermark(audio, sample_rate=K.S3GEN_SR)

    # -- decode -------------------------------------------------------------
    def group_batches(self, requests: list[AcousticRequest]) -> list[list[int]]:
        """Group rows that may legally share one forward pass.

        Groups are capped at ``max_batch_rows``: the flow solver's cost grows
        with padded length x rows, so an unbounded group would let one very
        long utterance stall everything batched with it.
        """
        groups: dict[tuple, list[int]] = {}
        for i, req in enumerate(requests):
            groups.setdefault(req.batch_key(self.ragged_batching), []).append(i)
        out: list[list[int]] = []
        for members in groups.values():
            for start in range(0, len(members), self.max_batch_rows):
                out.append(members[start : start + self.max_batch_rows])
        return out

    @torch.inference_mode()
    def _emit_stream_slice(self, req: AcousticRequest, audio: torch.Tensor) -> torch.Tensor:
        """Return only the samples this chunk may send, joined to the last one.

        ``audio`` is the whole prefix re-decoded at this chunk's length, so it
        overlaps everything already sent. Two rules make that safe:

        * **Emit once.** Only samples past what this request has already
          committed are returned; the rest is dropped, never re-sent.
        * **Hold back the join.** The final ``crossfade`` samples of each
          non-final chunk are retained instead of sent, and blended with the
          NEXT decode's version of that same region. Re-decoding a prefix at a
          greater length is a slightly different (equally valid) rendering, so
          butt-joining two decodes would step; blending them does not, and it
          costs one crossfade of added latency rather than a rewrite of audio
          the caller already has.
        """
        state = self._stream_state.get(req.request_id)
        committed = int(state["committed"]) if state else 0
        tail: torch.Tensor | None = state.get("tail") if state else None
        xf = int(self.stream_crossfade_samples)
        available = int(audio.shape[0])
        cut = available if req.finalize else available - xf

        if cut <= committed:
            # Not enough new audio to clear the held-back region yet.
            if not req.finalize:
                return audio.new_zeros(0)
            cut = available

        if tail is None:
            out = audio[:cut]
        else:
            n = min(xf, max(0, cut - committed), max(0, available - committed))
            if n <= 0:
                out = audio[committed:cut]
            else:
                ramp = torch.linspace(0.0, 1.0, n, device=audio.device, dtype=audio.dtype)
                blend = tail[:n] * (1.0 - ramp) + audio[committed : committed + n] * ramp
                out = torch.cat([blend, audio[committed + n : cut]])

        if req.finalize:
            self._stream_state.pop(req.request_id, None)
        else:
            self._stream_state[req.request_id] = {
                "committed": cut,
                "tail": audio[cut:available].clone(),
            }
        return out

    def release_stream_state(self, request_ids) -> None:
        """Drop held-back streaming tails for finished or cancelled requests."""
        for rid in request_ids:
            self._stream_state.pop(str(rid), None)

    def _flow_noise(
        self, req: AcousticRequest, mel_len: int, device, dtype
    ) -> torch.Tensor:
        """Flow noise for one row, drawn from a FIXED per-request bank.

        Incremental streaming decodes the same prefix more than once, at growing
        lengths. ``torch.randn((80, L))`` fills row-major, so the value at mel
        position j depends on L -- the same prefix would get different noise on
        every chunk, and the crossfade would be blending two unrelated
        renderings. Drawing a bank at a fixed cap and slicing it makes position
        j's noise independent of how much has been generated so far, which is
        what ``CausalConditionalCFM.rand_noise`` does upstream (it is ``None``
        in this checkpoint, so the port supplies it).
        """
        gen = request_generator(device, derive_seed(req.request_id, req.seed, "flow"))
        if not req.streaming:
            # One-shot decode keeps the exact draw the non-streaming path was
            # qualified with. The bank would be an equally valid rendering, but
            # it is a DIFFERENT one, and that path is already pinned by gates.
            return torch.randn(
                (K.S3GEN_N_MELS, mel_len), generator=gen, device=device, dtype=dtype
            )
        cap = max(int(mel_len), int(K.MAX_SPEECH_TOKENS) * K.S3GEN_TOKEN_MEL_RATIO)
        bank = torch.randn((K.S3GEN_N_MELS, cap), generator=gen, device=device, dtype=dtype)
        return bank[:, :mel_len]

    def decode_batch(self, requests: list[AcousticRequest]) -> list[AcousticResult]:
        """Decode one compatible group in a single flow-solver pass.

        Rows carry different reference voices (and, only under the opt-in and
        measurably unsafe ``ragged_batching``, different lengths). What makes
        mixed voices safe is doing per row what the reference does for one row:

        * each row's tokens are ``concat(its own prompt, its own codes)`` and
          only then padded, so a short reference never leaves a gap between the
          prompt block and the generated codes;
        * each row's prompt mel goes at ITS OWN offset, and its output is cut
          at that offset -- the upstream code assumes one shared prompt length
          for the whole batch;
        * each row's speaker embedding is its own, never ``references[0]``
          broadcast over rows belonging to other callers;
        * each row's flow noise is drawn at the length it would have used
          ALONE and then placed into the padded tensor, because a draw over the
          padded shape would depend on who else is in the batch.

        The vocoder then runs per row on that row's exact-length mel, so
        padding cannot reach the waveform.
        """
        if not requests:
            return []
        keys = {r.batch_key(self.ragged_batching) for r in requests}
        if len(keys) != 1:
            raise AcousticError(
                "decode_batch received rows with different batch keys; group them with "
                f"group_batches() first (got {len(keys)} distinct keys)"
            )

        device = next(self.s3gen.parameters()).device
        dtype = next(self.s3gen.parameters()).dtype
        flow = self.s3gen.flow
        ratio = int(flow.token_mel_ratio)
        n_mels = int(flow.output_size)
        rows = len(requests)

        for req in requests:
            if req.codes.numel() and not bool(
                ((req.codes >= 0) & (req.codes < K.CODEC_VOCAB_SIZE)).all()
            ):
                raise AcousticError(f"{req.request_id}: codec ids outside 0..{K.CODEC_VOCAB_SIZE - 1}")

        # --- per-row geometry ------------------------------------------------
        conds_list, prompt_lens, code_lens = [], [], []
        for req in requests:
            cond = req.conditioning.to(device)
            prompt_token = cond.prompt_token.reshape(1, -1).to(torch.long)
            declared = int(cond.prompt_token_len.reshape(-1)[0].item())
            declared = max(0, min(declared, int(prompt_token.shape[1])))
            prompt_feat = cond.prompt_feat.reshape(1, -1, n_mels)
            if prompt_feat.shape[1] < declared * ratio:
                raise AcousticError(
                    f"{req.request_id}: reference has {prompt_feat.shape[1]} mel frames, "
                    f"needs {declared * ratio} for {declared} prompt codes"
                )
            conds_list.append((cond, prompt_token[:, :declared], prompt_feat[:, : declared * ratio]))
            prompt_lens.append(declared)
            code_lens.append(int(req.codes.shape[-1]))

        # A non-final chunk drops the last `pre_lookahead_len` codes: the token
        # encoder's PreLookaheadLayer needs that many future codes to produce a
        # settled output, so emitting them now would emit audio the next chunk
        # would have rendered differently. This is the upstream `finalize=False`
        # contract, applied here because this path packs the batch itself.
        finalize = bool(requests[0].finalize)
        lookahead = 0 if finalize else int(flow.pre_lookahead_len)
        total_lens = [p + c for p, c in zip(prompt_lens, code_lens)]
        max_tokens = max(total_lens)
        mel_lens = [(n - lookahead) * ratio for n in total_lens]
        max_mel = max(mel_lens)
        emit_code_lens = [c - lookahead for c in code_lens]
        if min(emit_code_lens) < K.ACOUSTIC_MIN_EMIT_CODES and not finalize:
            # Not yet enough codes past the encoder's lookahead to emit
            # anything. That is an ordinary early-stream state, not an error:
            # the next chunk carries these codes again (chunks are cumulative),
            # so nothing is lost by returning silence now.
            return [
                AcousticResult(
                    request_id=req.request_id,
                    audio=torch.zeros(0, dtype=torch.float32, device=device),
                    n_valid_codes=code_lens[i],
                    watermarked=False,
                )
                for i, req in enumerate(requests)
            ]

        # --- packed token ids: each row's own prompt then its own codes -------
        tokens = torch.zeros((rows, max_tokens), dtype=torch.long, device=device)
        for i, req in enumerate(requests):
            _, prompt_token, _ = conds_list[i]
            tokens[i, : prompt_lens[i]] = prompt_token[0]
            tokens[i, prompt_lens[i] : total_lens[i]] = req.codes.to(device=device, dtype=torch.long)
        token_lens = torch.tensor(total_lens, dtype=torch.long, device=device)

        # --- speaker embeddings, one per row ---------------------------------
        embedding = torch.cat(
            [c[0].embedding.reshape(1, -1).to(device=device, dtype=dtype) for c in conds_list], dim=0
        )
        embedding = torch.nn.functional.normalize(embedding, dim=1)
        embedding = flow.spk_embed_affine_layer(embedding)

        # --- token encoder ----------------------------------------------------
        from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.s3gen.utils.mask import (
            make_pad_mask,
        )

        mask = (~make_pad_mask(token_lens)).unsqueeze(-1).to(embedding)
        embedded = flow.input_embedding(tokens) * mask
        h, h_masks = flow.encoder(embedded, token_lens)
        h = flow.encoder_proj(h)
        if h.shape[1] < max_mel:
            raise AcousticError(
                f"token encoder produced {h.shape[1]} mel frames, expected at least {max_mel}"
            )
        h = h[:, :max_mel]

        # --- per-row prompt mel at its OWN offset -----------------------------
        conds = torch.zeros((rows, max_mel, n_mels), device=device, dtype=h.dtype)
        for i, (_, _, prompt_feat) in enumerate(conds_list):
            conds[i, : prompt_lens[i] * ratio] = prompt_feat[0].to(dtype=h.dtype)
        conds = conds.transpose(1, 2)

        mel_mask = (
            ~make_pad_mask(torch.tensor(mel_lens, dtype=torch.long, device=device), max_len=max_mel)
        ).unsqueeze(1).to(h)

        # --- per-row noise, drawn at the length the row would use ALONE -------
        mu = h.transpose(1, 2).contiguous()
        noise = torch.zeros_like(mu)
        for i, req in enumerate(requests):
            noise[i, :, : mel_lens[i]] = self._flow_noise(
                req, mel_lens[i], device, mu.dtype
            )

        feat = flow.decoder(
            mu=mu,
            mask=mel_mask,
            spks=embedding,
            cond=conds,
            n_timesteps=self.cfm_timesteps,
            noise=noise,
        )[0]

        # --- per-row output: cut at this row's own prompt offset --------------
        results: list[AcousticResult] = []
        fade = self.s3gen.trim_fade.to(device=device)
        for i, req in enumerate(requests):
            start_mel = prompt_lens[i] * ratio
            row_mel = feat[i : i + 1, :, start_mel : start_mel + emit_code_lens[i] * ratio]
            row_mel = row_mel.to(dtype=dtype)

            voc_gen = request_generator(device, derive_seed(req.request_id, req.seed, "vocoder"))
            wav, _ = self.s3gen.hift_inference(row_mel, None, generator=voc_gen)
            wav = wav.clone()
            wav[:, : fade.shape[0]] *= fade.to(wav.dtype)

            audio = wav[0].reshape(-1).float()
            n_codes = code_lens[i]
            if req.finalize:
                audio = audio[: final_sample_limit(n_codes)]
            if req.streaming:
                audio = self._emit_stream_slice(req, audio)
            # In streaming mode every emitted chunk is watermarked, not just
            # the last one -- marking only the final chunk would ship almost the
            # whole utterance unmarked.
            if self.apply_watermark and (req.finalize or req.streaming):
                marked = self.watermark(audio.cpu().numpy())
                audio = torch.from_numpy(np.asarray(marked, dtype=np.float32)).to(audio.device)
            results.append(
                AcousticResult(
                    request_id=req.request_id,
                    audio=audio.contiguous(),
                    n_valid_codes=n_codes,
                    watermarked=bool((req.finalize or req.streaming) and self.apply_watermark),
                )
            )
        return results

    @torch.inference_mode()
    def decode(self, requests: list[AcousticRequest]) -> list[AcousticResult]:
        """Decode any set of requests, grouping compatible rows automatically."""
        out: list[AcousticResult | None] = [None] * len(requests)
        for group in self.group_batches(requests):
            batch = [requests[i] for i in group]
            for row, res in enumerate(self.decode_batch(batch)):
                out[group[row]] = res
        missing = [i for i, r in enumerate(out) if r is None]
        if missing:
            raise AcousticError(f"acoustic decode produced no result for rows {missing}")
        return out  # type: ignore[return-value]


def validate_codec_sequence(codes) -> list[int]:
    """Reject anything the acoustic stage must not be asked to decode."""
    out: list[int] = []
    for token in codes:
        token = int(token)
        if not is_valid_codec_id(token):
            raise AcousticError(
                f"codec id {token} is outside 0..{K.CODEC_VOCAB_SIZE - 1}; "
                "a '< 6561' filter alone does not reject negatives"
            )
        out.append(token)
    return out
