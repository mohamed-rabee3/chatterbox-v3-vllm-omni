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


#: Set CBX_ACOUSTIC_PROFILE=1 to log where stage-1 wall time actually goes.
#: Off by default and it costs one `perf_counter` per phase when on; the
#: synchronisation it needs to attribute GPU time makes it a diagnostic, not a
#: production setting.
_PROFILE = bool(int(os.environ.get("CBX_ACOUSTIC_PROFILE", "0")))
_PROFILE_EVERY = float(os.environ.get("CBX_ACOUSTIC_PROFILE_SECS", "15"))


class _AcousticStats:
    """Rolling per-phase totals for the acoustic decode, logged periodically."""

    def __init__(self) -> None:
        import time as _t

        self._t = _t
        self.reset(_t.perf_counter())

    def reset(self, now: float) -> None:
        self.t0 = now
        self.calls = 0
        self.rows = 0
        self.codes = 0
        self.emitted_samples = 0
        self.phases = {"setup": 0.0, "flow": 0.0, "vocoder": 0.0,
                       "watermark": 0.0, "other": 0.0}
        self.eager_flow = 0
        self.eager_voc = 0

    def maybe_log(self) -> None:
        now = self._t.perf_counter()
        wall = now - self.t0
        if wall < _PROFILE_EVERY or not self.calls:
            return
        total = sum(self.phases.values())
        parts = " ".join(
            f"{k}={v * 1e3 / self.calls:.1f}ms({100.0 * v / max(total, 1e-9):.0f}%)"
            for k, v in self.phases.items()
        )
        logger.info(
            "[cbx-acoustic] %.0fs: %d batches, %.2f rows/batch, busy %.0f%% of wall | %s "
            "| audio_out %.2fx realtime | eager flow=%d voc=%d",
            wall, self.calls, self.rows / self.calls, 100.0 * total / wall, parts,
            self.emitted_samples / float(K.S3GEN_SR) / wall, self.eager_flow, self.eager_voc,
        )
        self.reset(now)


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
            # Equal generated lengths alone are insufficient when reference
            # clips have different lengths: the bidirectional encoder would
            # see padding for the shorter row.
            key.append(self.conditioning.acoustic_prompt_length())
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
        flow_cudagraphs: bool = False,
        compile_estimator: bool = False,
        estimator_dtype: str = "float32",
        graph_cache_size: int = 16,
        graph_max_codes: int = 75,
        prewarm_prompt_tokens: tuple[int, ...] = (),
        materialize_vocoder_weights: bool = False,
        vocoder_cudagraphs: bool = False,
        watermark_workers: int = K.ACOUSTIC_WATERMARK_WORKERS,
        watermark_device: str = K.ACOUSTIC_WATERMARK_DEVICE,
        batch_vocoder: bool = False,
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
        self.flow_cudagraphs = bool(flow_cudagraphs)
        self.compile_estimator = bool(compile_estimator)
        if estimator_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"unsupported acoustic estimator dtype: {estimator_dtype}")
        self.estimator_dtype = getattr(torch, estimator_dtype)
        self.graph_cache_size = max(1, int(graph_cache_size))
        self.graph_max_codes = max(5, int(graph_max_codes))
        self.prewarm_prompt_tokens = tuple(int(n) for n in prewarm_prompt_tokens)
        self.materialize_vocoder_weights = bool(materialize_vocoder_weights)
        self.vocoder_cudagraphs = bool(vocoder_cudagraphs)
        self._flow_graphs = {}
        self._flow_graph_pool = None
        self._watermarker = None
        self.watermark_workers = max(1, int(watermark_workers))
        if watermark_device not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"unsupported acoustic watermark device: {watermark_device}")
        self.watermark_device = watermark_device
        self._wm_pool = None
        self.batch_vocoder = bool(batch_vocoder)
        self._stats = _AcousticStats() if _PROFILE else None
        #: Held-back tail per streaming request. Streaming can never revise what
        #: it already sent, so the overlap is retained here and blended with the
        #: next chunk's decode of the same region before being emitted.
        self._stream_state: dict[str, dict] = {}
        self._stream_noise: dict[str, tuple[tuple, torch.Tensor]] = {}
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
        # Legacy torch.Tensor(...) constructors in relative attention create
        # CPU parameters even inside vLLM's CUDA device context. Move the whole
        # acoustic module to the embedding's execution device once; otherwise
        # every attention layer copies its position bias during every decode.
        self.s3gen.to(device=self.s3gen.flow.input_embedding.weight.device)
        self.s3gen.flow.decoder.estimator.to(dtype=self.estimator_dtype)
        self.s3gen.flow.decoder.fp32_solver = self.estimator_dtype != torch.float32
        if self.materialize_vocoder_weights:
            from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
            for module in list(self.s3gen.mel2wav.modules()):
                if is_parametrized(module, "weight"):
                    remove_parametrizations(module, "weight", leave_parametrized=True)
        logger.info(
            "Chatterbox MTL V3 S3Gen: loaded %s (profile %s), %d tensors",
            filename, self.checkpoint_profile, len(state),
        )

    @torch.inference_mode()
    def warm_compiled_estimator(self):
        """Compile before readiness, with fresh noise still supplied per request."""
        if not self.compile_estimator:
            return
        torch.set_float32_matmul_precision("highest")
        self.s3gen.eval()
        decoder = self.s3gen.flow.decoder
        decoder.estimator = torch.compile(
            decoder.estimator, dynamic=True, mode="max-autotune-no-cudagraphs",
        )
        device = self.s3gen.flow.input_embedding.weight.device
        dtype = self.estimator_dtype
        z = torch.zeros((2, 80, 504), device=device, dtype=dtype)
        decoder.estimator.forward(
            x=z, mask=torch.ones((2, 1, 504), device=device, dtype=dtype),
            mu=z.clone(), t=torch.zeros(2, device=device, dtype=dtype),
            spks=torch.zeros((2, 80), device=device, dtype=dtype), cond=z.clone(), r=None,
        )
        torch.cuda.synchronize(device)
        logger.info("Chatterbox compiled acoustic estimator warmed: %s", self.estimator_dtype)
        self.warm_acoustic_graphs()

    @torch.inference_mode()
    def warm_acoustic_graphs(self):
        """Capture configured reference geometries before admitting traffic.

        All static tensors are overwritten on replay. No voice, noise bank or
        request metadata from this synthetic warmup is retained as input.
        """
        if not self.flow_cudagraphs or not self.prewarm_prompt_tokens:
            return
        device = self.s3gen.flow.input_embedding.weight.device
        # Warm the rungs THIS deploy's ladder actually decodes at, not a fixed
        # list: a shape that is not captured here is captured during the first
        # requests that hit it, and a cold server measures materially worse than
        # a warm one (20 callers, same profile: p95 buffered stall 1.184 s cold
        # against 0.588 s warm). The ladder is the same one stage 0 transports
        # on, so these are exactly the lengths that will arrive.
        rungs, cum, block = [], 0, K.ACOUSTIC_STREAM_FIRST_BLOCK
        while cum < 4 * K.ACOUSTIC_STREAM_MAX_BLOCK and len(rungs) < 12:
            cum += block
            rungs.append(cum)
            block = min(int(block * K.ACOUSTIC_STREAM_BLOCK_GROWTH),
                        K.ACOUSTIC_STREAM_MAX_BLOCK)
        for prompt in self.prewarm_prompt_tokens:
            if prompt <= 0:
                raise ValueError("prewarm prompt lengths must be positive")
            for rows in (1, 2, 3, 4):
                for codes in rungs:
                    if codes > self.graph_max_codes:
                        continue
                    length = prompt + codes
                    frames = (length - K.ACOUSTIC_PRE_LOOKAHEAD_LEN) * K.S3GEN_TOKEN_MEL_RATIO
                    self._run_acoustic_core(
                        torch.zeros((rows, length), dtype=torch.long, device=device),
                        torch.full((rows,), length, dtype=torch.long, device=device),
                        torch.zeros((rows, self.s3gen.flow.spk_embed_affine_layer.in_features), device=device),
                        torch.zeros((rows, K.S3GEN_N_MELS, frames), device=device),
                        torch.zeros((rows, K.S3GEN_N_MELS, frames), device=device),
                        torch.ones((rows, 1, frames), device=device), cacheable=True,
                    )
        torch.cuda.synchronize(device)
        logger.info("Chatterbox acoustic graphs ready before admission: %d", len(self._flow_graphs))

    def _run_vocoder(self, mel, generator, *, cacheable=False):
        if not self.vocoder_cudagraphs:
            return self.s3gen.hift_inference(mel, None, generator=generator)[0]
        v = self.s3gen.mel2wav
        f0 = v.f0_predictor(mel)
        source = v.f0_upsamp(f0[:, None]).transpose(1, 2)
        source, _, _ = v.m_source(source, generator)
        real, imaginary = v._stft(source.transpose(1, 2).squeeze(1))
        spectrum = torch.cat([real, imaginary], dim=1)
        def run(values):
            return torch.stack(v.decode_spectral(*values))
        magnitude, phase = self._run_graph((mel, spectrum), run,
                                           cacheable=cacheable, namespace="vocoder")
        return v._istft(magnitude, phase).clamp(-v.audio_limit, v.audio_limit)

    # -- watermark ----------------------------------------------------------
    def _watermark_run_device(self) -> str:
        """Where Perth's DSP runs. Not a quality knob -- see below."""
        want = self.watermark_device
        if want == "auto":
            want = "cuda" if torch.cuda.is_available() else "cpu"
        if want == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return want

    def _get_watermarker(self):
        if self._watermarker is None:
            import perth

            device = self._watermark_run_device()
            # Perth defaults to CPU, and at 40-140 ms per emitted chunk that
            # single-threaded DSP -- not the flow solver -- was the ceiling on
            # sustained streaming throughput: it runs inside the stage-1
            # forward, so the GPU idles for its whole duration on every chunk
            # of every stream. Everything after the resample is plain torch
            # (an STFT pair around a 2.4 M-parameter conv encoder), so it runs
            # on the acoustic device unchanged.
            #
            # This is a placement change, not a model change. The watermark is
            # the same network with the same weights, the resampler that
            # defines the signal band stays on CPU (librosa/soxr, bit-identical),
            # and measured against the CPU run the output differs by at most
            # 7e-6 -- about 1e-4 of the watermark's own perturbation of the
            # waveform, and far below this port's accepted acoustic variation.
            self._watermarker = perth.PerthImplicitWatermarker(device=device)
            logger.info("Chatterbox Perth watermarker running on %s", device)
        return self._watermarker

    def watermark(self, audio: np.ndarray) -> np.ndarray:
        """Reference Perth watermark. Never silently skipped to look faster."""
        if not self.apply_watermark:
            return audio
        return self._get_watermarker().apply_watermark(audio, sample_rate=K.S3GEN_SR)

    @staticmethod
    def _watermarkable(audio: np.ndarray) -> bool:
        # An emitted chunk can legitimately be empty -- the crossfade holdback
        # can consume a short chunk entirely, and a resumed stream re-decodes a
        # region it has already sent. Perth's STFT reshapes to (-1, n) and dies
        # on a zero-length signal, so an empty chunk is passed through unmarked
        # rather than taking the acoustic engine down. There is no audio in it
        # to mark.
        return audio.size > 0

    def _watermark_one(self, audio: np.ndarray) -> np.ndarray:
        if not self._watermarkable(audio):
            return audio
        # Workers run outside the caller's `inference_mode` region (the mode is
        # thread-local), so re-enter it here: that is the context the serial
        # path ran under, and it keeps autograd off the watermark encoder.
        with torch.inference_mode():
            return self._get_watermarker().apply_watermark(audio, sample_rate=K.S3GEN_SR)

    def watermark_many(self, audios: list[np.ndarray]) -> list[np.ndarray]:
        """Watermark a decode batch's rows concurrently.

        Perth is a CPU model (a 2.4 M-parameter encoder around an STFT pair) and
        costs 40-140 ms per row, which is comparable to the whole GPU acoustic
        decode. Run serially inside the stage-1 forward -- which is where the
        reference loop put it -- it leaves the GPU idle for rows x 40-140 ms on
        every batch, and that, not the flow solver, is what caps sustained
        streaming throughput.

        Each row is still passed through the *same* `apply_watermark` call with
        the same input, so this is a scheduling change, not a numerical one: the
        outputs are bit-identical to the serial path (`torch` releases the GIL
        in these kernels, so the threads genuinely overlap).
        """
        if not self.apply_watermark or not audios:
            return audios
        if len(audios) == 1:
            return [self._watermark_one(audios[0])]
        self._get_watermarker()  # build/warm once, never inside a worker
        if self._watermark_run_device() != "cpu":
            # On the acoustic device a row costs 2-5 ms and the rows already
            # serialise on that device; threads would only add contention.
            return [self._watermark_one(a) for a in audios]
        return list(self._watermark_pool().map(self._watermark_one, audios))

    def _watermark_pool(self) -> "ThreadPoolExecutor":
        if self._wm_pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._wm_pool = ThreadPoolExecutor(
                max_workers=self.watermark_workers, thread_name_prefix="cbx-perth"
            )
        return self._wm_pool

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

        ``committed``/``cut``/``available`` are ABSOLUTE PCM sample coordinates
        for the whole utterance. When ``req.token_offset`` is non-zero the
        bounded-window decode only rendered codes ``[token_offset:]``, so
        ``audio`` starts at absolute sample ``base``; every index into it is
        shifted by ``base``. ``token_offset == 0`` (window disabled, or the
        finalize chunk) makes ``base == 0`` and this identical to the original.
        """
        base = int(getattr(req, "token_offset", 0)) * K.SAMPLES_PER_CODEC_TOKEN
        state = self._stream_state.get(req.request_id)
        committed = int(state["committed"]) if state else 0
        tail: torch.Tensor | None = state.get("tail") if state else None
        xf = int(self.stream_crossfade_samples)
        local_len = int(audio.shape[0])
        available = base + local_len
        cut = available if req.finalize else available - xf

        if cut <= committed:
            # Not enough new audio to clear the held-back region yet.
            if not req.finalize:
                return audio.new_zeros(0)
            cut = available

        if base > committed:
            # The window must cover everything already emitted; otherwise the
            # samples in (committed, base) were never rendered and the stream
            # would have a gap. Widen ACOUSTIC_STREAM_CTX_WINDOW or lower the
            # block growth so consecutive windows overlap.
            raise AcousticError(
                f"{req.request_id}: streaming context window starts at sample "
                f"{base} but {committed} samples are already committed"
            )

        if tail is None:
            out = audio[: cut - base]
        else:
            n = min(xf, max(0, cut - committed), max(0, available - committed))
            if n <= 0:
                out = audio[committed - base : cut - base]
            else:
                ramp = torch.linspace(0.0, 1.0, n, device=audio.device, dtype=audio.dtype)
                blend = (
                    tail[:n] * (1.0 - ramp)
                    + audio[committed - base : committed - base + n] * ramp
                )
                out = torch.cat([blend, audio[committed - base + n : cut - base]])

        if req.finalize:
            self._stream_state.pop(req.request_id, None)
            self._stream_noise.pop(req.request_id, None)
        else:
            self._stream_state[req.request_id] = {
                "committed": cut,
                "tail": audio[cut - base : available - base].clone(),
            }
        return out

    def release_stream_state(self, request_ids) -> None:
        """Drop held-back streaming tails for finished or cancelled requests."""
        for rid in request_ids:
            self._stream_state.pop(str(rid), None)
            self._stream_noise.pop(str(rid), None)

    def _flow_noise(
        self, req: AcousticRequest, mel_len: int, device, dtype, *, prompt_mel: int = 0
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

        With a bounded context window (``req.token_offset > 0``) the decoded
        sequence is ``[prompt] + codes[token_offset:]``: the prompt keeps bank
        positions ``[0:prompt_mel]`` but the generated region must read from the
        ABSOLUTE bank offset for its first code, so generated code j still gets
        ``bank[:, prompt_mel + j*ratio]`` regardless of where the window starts.
        """
        gen = request_generator(device, derive_seed(req.request_id, req.seed, "flow"))
        if not req.streaming:
            # One-shot decode keeps the exact draw the non-streaming path was
            # qualified with. The bank would be an equally valid rendering, but
            # it is a DIFFERENT one, and that path is already pinned by gates.
            return torch.randn(
                (K.S3GEN_N_MELS, mel_len), generator=gen, device=device, dtype=dtype
            )
        ratio = K.S3GEN_TOKEN_MEL_RATIO
        token_offset = int(getattr(req, "token_offset", 0))
        # Quantise the bank size so a growing window never changes ``identity``
        # and forces a redraw (which would break noise continuity mid-stream).
        unit = int(K.MAX_SPEECH_TOKENS) * ratio
        need = int(mel_len) + token_offset * ratio
        cap = ((max(0, need - 1) // unit) + 1) * unit
        identity = (req.seed, cap, device, dtype)
        cached = self._stream_noise.get(req.request_id)
        bank = None
        if cached is not None and cached[0] == identity:
            bank = cached[1]
        if bank is None:
            bank = torch.randn((K.S3GEN_N_MELS, cap), generator=gen, device=device, dtype=dtype)
            self._stream_noise[req.request_id] = (identity, bank)
        if token_offset == 0:
            return bank[:, :mel_len]
        gen_off = int(prompt_mel) + token_offset * ratio
        gen_len = int(mel_len) - int(prompt_mel)
        return torch.cat(
            [bank[:, : int(prompt_mel)], bank[:, gen_off : gen_off + gen_len]], dim=1
        )

    def _phase_timer(self):
        """Return a `mark(phase)` that attributes elapsed time to that phase.

        A no-op unless CBX_ACOUSTIC_PROFILE is set. When on it synchronises the
        device so GPU phases are not credited to whichever CPU phase happens to
        touch the result first -- accurate, but not free.
        """
        stats = self._stats
        if stats is None:
            return lambda _phase: None

        import time as _t

        last = [_t.perf_counter()]

        def mark(phase: str) -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now = _t.perf_counter()
            stats.phases[phase] = stats.phases.get(phase, 0.0) + (now - last[0])
            last[0] = now

        return mark

    def _run_graph(self, args, run, *, cacheable=False, namespace="flow"):
        """Replay an exact-shape tensor computation with fresh request inputs.

        No padding or request metadata is captured. Only recurrent streaming
        rungs are cached, with a hard cap; arbitrary final lengths stay eager.
        The shared pool is safe because this runner executes serially and the
        result is cloned before any other graph can reuse its storage.
        """
        mu = args[0]
        if not (self.flow_cudagraphs and mu.is_cuda and cacheable):
            return run(args)
        key = (namespace, self.cfm_timesteps, tuple((tuple(t.shape), t.dtype, t.device) for t in args))
        cached = self._flow_graphs.get(key)
        if cached is None:
            if len(self._flow_graphs) >= getattr(self, "graph_cache_size", 16):
                # Cache full: this shape runs eager for the rest of the process.
                # Counted, because a cache too small for the shapes the ladder
                # actually produces silently turns graph replay off under load.
                if self._stats is not None:
                    if namespace == "flow":
                        self._stats.eager_flow += 1
                    else:
                        self._stats.eager_voc += 1
                return run(args)
            static = tuple(t.clone().contiguous() for t in args)
            current = torch.cuda.current_stream(mu.device)
            capture_stream = torch.cuda.Stream(device=mu.device)
            capture_stream.wait_stream(current)
            with torch.cuda.stream(capture_stream):
                for _ in range(2):
                    run(static)
            current.wait_stream(capture_stream)
            graph = torch.cuda.CUDAGraph()
            if self._flow_graph_pool is None:
                self._flow_graph_pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph, pool=self._flow_graph_pool, stream=capture_stream):
                output = run(static)
            cached = (graph, static, output)
            self._flow_graphs[key] = cached
            logger.info("Chatterbox acoustic %s CUDA graph: rows=%d length=%d (%d/%d)",
                        namespace, mu.shape[0], mu.shape[-1], len(self._flow_graphs),
                        getattr(self, "graph_cache_size", 16))
        graph, static, output = cached
        for dst, src in zip(static, args):
            dst.copy_(src)
        graph.replay()
        return output.clone()

    def _run_flow(self, mu, mask, embedding, conds, noise, *, cacheable=False):
        def run(values):
            m, mask_, speaker, cond, z = values
            return self.s3gen.flow.decoder(
                mu=m, mask=mask_, spks=speaker, cond=cond,
                n_timesteps=self.cfm_timesteps, noise=z,
            )[0]
        return self._run_graph((mu, mask, embedding, conds, noise), run,
                               cacheable=cacheable)

    def _run_acoustic_core(self, tokens, token_lens, embedding, conds, noise,
                           mel_mask, *, cacheable=False):
        """Capture token encoder and flow together; reference geometry stays outside."""
        from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.s3gen.utils.mask import make_pad_mask
        flow = self.s3gen.flow
        def run(values):
            tok, lengths, speaker, cond, z, mask_mel = values
            speaker = torch.nn.functional.normalize(speaker, dim=1)
            speaker = flow.spk_embed_affine_layer(speaker)
            mask = (~make_pad_mask(lengths, tok.shape[1])).unsqueeze(-1).to(speaker)
            embedded = flow.input_embedding(tok) * mask
            h, _ = flow.encoder(embedded, lengths)
            h = flow.encoder_proj(h)[:, :z.shape[-1]]
            return flow.decoder(
                mu=h.transpose(1, 2).contiguous(), mask=mask_mel,
                spks=speaker, cond=cond, noise=z,
                n_timesteps=self.cfm_timesteps,
            )[0]
        return self._run_graph((tokens, token_lens, embedding, conds, noise, mel_mask),
                               run, cacheable=cacheable, namespace="encoder_flow")

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
            declared = cond.acoustic_prompt_length()
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
        from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.s3gen.utils.mask import make_pad_mask

        # --- per-row prompt mel at its OWN offset -----------------------------
        conds = torch.zeros((rows, max_mel, n_mels), device=device, dtype=dtype)
        for i, (_, _, prompt_feat) in enumerate(conds_list):
            conds[i, : prompt_lens[i] * ratio] = prompt_feat[0].to(dtype=dtype)
        conds = conds.transpose(1, 2)

        mel_mask = (
            ~make_pad_mask(torch.tensor(mel_lens, dtype=torch.long, device=device), max_len=max_mel)
        ).unsqueeze(1).to(device=device, dtype=dtype)

        # --- per-row noise, drawn at the length the row would use ALONE -------
        noise = torch.zeros((rows, n_mels, max_mel), device=device, dtype=dtype)
        for i, req in enumerate(requests):
            noise[i, :, : mel_lens[i]] = self._flow_noise(
                req, mel_lens[i], device, dtype, prompt_mel=prompt_lens[i] * ratio
            )

        _mark = self._phase_timer()
        _mark("setup")
        feat = self._run_acoustic_core(
            tokens, token_lens, embedding, conds, noise, mel_mask,
            cacheable=(not finalize and bool(requests[0].streaming)
                       and max(code_lens) <= self.graph_max_codes and rows <= 4),
        )

        _mark("flow")

        # --- per-row output: cut at this row's own prompt offset --------------
        results: list[AcousticResult] = []
        row_audio: list[torch.Tensor] = []
        row_marked: list[bool] = []
        row_codes: list[int] = []
        fade = self.s3gen.trim_fade.to(device=device)
        # --- vocoder ---------------------------------------------------------
        # Every row in a group shares its code length AND its prompt length --
        # both are part of ``batch_key`` -- so all rows' mel slices have the
        # same offset and width, and the vocoder can run once for the batch
        # instead of once per row. The flow solver was already batched; leaving
        # the vocoder in a Python loop meant the second half of the acoustic
        # stage was serialised over the group, which is what the batching was
        # meant to remove.
        #
        # The per-row RNG is preserved exactly: ``_draw`` accepts one generator
        # per row and draws that row's noise at the row's own shape, taking the
        # same values in the same order from the same stream as the solo call.
        voc_gens = [
            request_generator(device, derive_seed(req.request_id, req.seed, "vocoder"))
            for req in requests
        ]
        # `ragged_batching` (opt-in, and measurably unsafe on this checkpoint)
        # packs unequal lengths, so the shared slice below would not hold. Check
        # the geometry rather than trusting the batch key.
        uniform_geometry = (
            len(set(prompt_lens)) == 1 and len(set(emit_code_lens)) == 1
        )
        batched_wavs = None
        if self.batch_vocoder and rows > 1 and uniform_geometry:
            start_mel = prompt_lens[0] * ratio
            width = emit_code_lens[0] * ratio
            batch_mel = feat[:, :, start_mel : start_mel + width].to(dtype=dtype)
            batched_wavs = self._run_vocoder(
                batch_mel, voc_gens,
                cacheable=(requests[0].streaming and not finalize
                           and code_lens[0] <= self.graph_max_codes),
            )

        for i, req in enumerate(requests):
            if batched_wavs is not None:
                wav = batched_wavs[i : i + 1]
            else:
                start_mel = prompt_lens[i] * ratio
                row_mel = feat[i : i + 1, :, start_mel : start_mel + emit_code_lens[i] * ratio]
                row_mel = row_mel.to(dtype=dtype)
                wav = self._run_vocoder(row_mel, voc_gens[i], cacheable=(req.streaming
                    and not req.finalize and code_lens[i] <= self.graph_max_codes))
            wav = wav.clone()
            # The initial trim fade belongs only to the true utterance start. A
            # bounded-window chunk (token_offset > 0) renders a mid-utterance
            # slice, so fading its first samples would notch audio the previous
            # chunk already delivered cleanly.
            if int(getattr(req, "token_offset", 0)) == 0:
                wav[:, : fade.shape[0]] *= fade.to(wav.dtype)

            audio = wav[0].reshape(-1).float()
            n_codes = code_lens[i]
            if req.finalize:
                audio = audio[: final_sample_limit(n_codes)]
            if req.streaming:
                audio = self._emit_stream_slice(req, audio)
            # In streaming mode every emitted chunk is watermarked, not just
            # the last one -- marking only the final chunk would ship almost the
            # whole utterance unmarked. The rows are collected first and marked
            # together below: the call per row is identical, but a batch's rows
            # no longer wait on each other's CPU time with the GPU idle.
            row_audio.append(audio)
            row_marked.append(bool((req.finalize or req.streaming) and self.apply_watermark))
            row_codes.append(n_codes)

        _mark("vocoder")
        marked_idx = [i for i, m in enumerate(row_marked) if m and row_audio[i].numel()]
        for i, m in enumerate(row_marked):
            row_marked[i] = bool(m and row_audio[i].numel())
        if marked_idx:
            marked = self.watermark_many([row_audio[i].cpu().numpy() for i in marked_idx])
            for i, out in zip(marked_idx, marked):
                row_audio[i] = torch.from_numpy(np.asarray(out, dtype=np.float32)).to(
                    row_audio[i].device
                )

        _mark("watermark")

        for i, req in enumerate(requests):
            results.append(
                AcousticResult(
                    request_id=req.request_id,
                    audio=row_audio[i].contiguous(),
                    n_valid_codes=row_codes[i],
                    watermarked=row_marked[i],
                )
            )
        _mark("other")
        if self._stats is not None:
            st = self._stats
            st.calls += 1
            st.rows += rows
            st.codes += sum(code_lens)
            st.emitted_samples += sum(int(a.numel()) for a in row_audio)
            st.maybe_log()
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
