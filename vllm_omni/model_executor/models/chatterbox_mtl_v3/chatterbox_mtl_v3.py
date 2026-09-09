# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox Multilingual V3 as a two-stage native vLLM-Omni model.

Stage 0 ``chatterbox_mtl_v3_t3``    (``LLM_AR``)         text + conditioning -> codec ids
Stage 1 ``chatterbox_mtl_v3_s3gen`` (``LLM_GENERATION``) codec ids -> 24 kHz waveform

Prompt layout for stage 0 (both CFG rows carry the SAME token ids)::

    [COND placeholder] * 34 | text ids (SOT .. EOT) | speech BOS | speech BOS

Only the 34 conditioning positions are multimodal. Everything else is a real
token id, and the embedding a row gets is decided by its **global position**
inside its own request, not by its id -- a codec id and a text id are
indistinguishable as integers, so id-based dispatch would be wrong.

Because ``embed_input_ids`` sees no positions, it only fills the multimodal
block; :meth:`ChatterboxMTLV3T3.forward` then applies the text/BOS/decode
content and the two learned position tables using ``positions`` and each
request's own ``prompt_len``/``text_len``. That also means the learned speech
position of a decoded token is derived from the request's own progress, which
survives slot compaction, preemption and recomputation.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.data_entry_keys import EmbeddingsStruct, OmniPayloadStruct, to_dict, to_struct
from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K
from vllm_omni.model_executor.models.chatterbox_mtl_v3.conditioning import (
    ConditioningCache,
    ConditioningEncoder,
    ReferenceConditioning,
    audio_content_hash,
    conditioning_cache_key,
)
from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import AcousticRequest, ChatterboxS3Gen
from vllm_omni.model_executor.models.chatterbox_mtl_v3.t3 import ChatterboxT3, backbone_vllm_config
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.chatterbox_mtl_v3 import ChatterboxMTLV3Config
from vllm_omni.transformers_utils.repo_utils import hf_api

logger = init_logger(__name__)

# Set CBX_STREAM_DEBUG=1 to trace every acoustic chunk (codes in, samples out).
_STREAM_DEBUG = bool(int(os.environ.get("CBX_STREAM_DEBUG", "0")))


class AcousticPayloadError(RuntimeError):
    """A stage-1 payload that cannot be turned into a valid reference voice."""

#: Filler token id for the 34 conditioning placeholder positions. Never embedded
#: (those rows always come from the multimodal block), but it must be a valid
#: index so a stray gather cannot read out of range.
COND_PLACEHOLDER_ID = 0

_ENCODER_CACHE: dict[str, ConditioningEncoder] = {}
_CONDITIONING_CACHE: dict[str, ConditioningCache] = {}


def resolve_model_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    return hf_api().snapshot_download(model)


def _get_encoder(model_dir: str, profile: str) -> ConditioningEncoder:
    """Process-wide conditioning encoder.

    The multimodal processor is rebuilt per request, so without this every
    request would reload ``ve.pt`` and the S3Gen tokenizer/speaker encoder --
    hundreds of milliseconds on the TTFA critical path.
    """
    key = f"{model_dir}::{profile}"
    enc = _ENCODER_CACHE.get(key)
    if enc is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        enc = ConditioningEncoder(model_dir, device=device, checkpoint_profile=profile)
        _ENCODER_CACHE[key] = enc
    return enc


def _get_conditioning_cache(model_dir: str, capacity: int) -> ConditioningCache:
    cache = _CONDITIONING_CACHE.get(model_dir)
    if cache is None:
        cache = ConditioningCache(capacity=capacity)
        _CONDITIONING_CACHE[model_dir] = cache
    return cache


class ChatterboxMTLV3ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> ChatterboxMTLV3Config:
        return self.ctx.get_hf_config(ChatterboxMTLV3Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Exactly one reference voice per request.
        return {"audio": 1}

    def get_data_parser(self):
        """Deliver reference audio as mono 24 kHz, resampled the reference way.

        The reference wrapper loads with ``librosa.load(..., sr=24000)``, whose
        default resampler is ``soxr_hq``; ``soxr`` here is the same kernel.
        vLLM's default (``pyav``) is a different resampler, and the reference
        signal is what every downstream conditioning window is cut from, so the
        resampler choice is part of the voice.
        """
        from vllm.multimodal.parse import MultiModalDataParser

        return MultiModalDataParser(
            target_sr=float(K.S3GEN_SR),
            target_channels=1,
            audio_resample_method="soxr",
        )


class ChatterboxMTLV3MultiModalProcessor(BaseMultiModalProcessor[ChatterboxMTLV3ProcessingInfo]):
    """Turns (text, language, reference audio) into prompt ids + conditioning.

    Runs in the input-processing process. Reference conditioning is
    content-addressed and cached here, so a repeated voice costs a dict lookup
    instead of a voice-encoder + tokenizer + speaker-encoder pass.
    """

    def _tokenizer(self, model_dir: str):
        tok = getattr(self, "_mtl_tokenizer", None)
        if tok is None:
            from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.tokenizers import (
                MTLTokenizer,
            )

            tok = MTLTokenizer(os.path.join(model_dir, K.TOKENIZER_FILE))
            self._mtl_tokenizer = tok
        return tok

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        from vllm_omni.model_executor.models.chatterbox_mtl_v3.text import (
            build_text_token_ids,
            normalize_language,
        )

        config = self.info.ctx.get_hf_config(ChatterboxMTLV3Config)
        model_dir = resolve_model_dir(self.info.ctx.model_config.model)
        tokenizer = self._tokenizer(model_dir)

        language = normalize_language(str(mm_kwargs.get("language") or "en"), config)
        text_ids = build_text_token_ids(tokenizer, prompt, language, config)
        # Two BOS embeddings terminate the prefill; see the module docstring.
        input_ids = torch.tensor(
            list(text_ids) + [K.START_SPEECH_TOKEN] * K.NUM_PREFILL_BOS, dtype=torch.long
        ).unsqueeze(0)

        audio = mm_data.get("audio")
        if audio is None:
            audios = mm_data.get("audios")
            audio = (audios[0], K.S3GEN_SR) if audios else None
        if audio is None:
            # Profiling / warmup path: no voice, so no conditioning.
            return BatchFeature({"input_ids": input_ids, "text_len": [len(text_ids)]})

        wav, sr = audio
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        exaggeration = float(mm_kwargs.get("exaggeration", config.default_exaggeration))

        key = conditioning_cache_key(
            audio_content_hash=audio_content_hash(wav, int(sr)),
            checkpoint_profile=config.checkpoint_profile,
        )
        cache = _get_conditioning_cache(model_dir, config.conditioning_cache_size)
        cond = cache.get(key)
        if cond is None:
            encoder = _get_encoder(model_dir, config.checkpoint_profile)
            cond = encoder.encode(wav, int(sr), max_seconds=config.max_reference_seconds)
            cond = cache.put(cond)

        return BatchFeature(
            {
                "input_ids": input_ids,
                "text_len": [len(text_ids)],
                "speaker_emb": cond.speaker_emb.detach().cpu(),
                "cond_prompt_speech_tokens": cond.cond_prompt_speech_tokens.detach().cpu(),
                # Collation right-pads shorter speech prompts; carry the real
                # length so padding never reaches the Perceiver.
                "cond_prompt_len": torch.tensor(
                    [int(cond.cond_prompt_speech_tokens.shape[-1])], dtype=torch.long
                ),
                "exaggeration": torch.tensor([[[exaggeration]]], dtype=torch.float32),
                "speech_token": cond.prompt_token.detach().cpu(),
                "speech_token_len": [cond.prompt_token_len.detach().cpu()],
                "speech_feat": cond.prompt_feat.detach().cpu(),
                "embedding": cond.embedding.detach().cpu(),
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "speaker_emb": MultiModalFieldConfig.batched("audio"),
            "cond_prompt_speech_tokens": MultiModalFieldConfig.batched("audio"),
            "cond_prompt_len": MultiModalFieldConfig.batched("audio"),
            "exaggeration": MultiModalFieldConfig.batched("audio"),
            "speech_token": MultiModalFieldConfig.batched("audio"),
            "speech_token_len": MultiModalFieldConfig.batched("audio"),
            "speech_feat": MultiModalFieldConfig.batched("audio"),
            "embedding": MultiModalFieldConfig.batched("audio"),
        }

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # A fixed-size block: 1 speaker projection + 32 perceiver outputs +
        # 1 exaggeration projection.
        return [
            PromptInsertion(
                modality="audio",
                target=PromptIndexTargets.start(),
                insertion=[COND_PLACEHOLDER_ID] * K.COND_PREFIX_LEN,
            )
        ]


class ChatterboxMTLV3DummyInputsBuilder(BaseDummyInputsBuilder[ChatterboxMTLV3ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "Hello, this is a warmup utterance for the Chatterbox multilingual model."

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 1)
        overrides = mm_options.get("audio") if mm_options else None
        length = K.DEC_COND_SECONDS * K.S3GEN_SR
        return {
            "audio": (
                self._get_dummy_audios(length=length, num_audios=num_audios, overrides=overrides)[0],
                K.S3GEN_SR,
            )
        }


@MULTIMODAL_REGISTRY.register_processor(
    ChatterboxMTLV3MultiModalProcessor,
    info=ChatterboxMTLV3ProcessingInfo,
    dummy_inputs=ChatterboxMTLV3DummyInputsBuilder,
)
class ChatterboxMTLV3T3(nn.Module, SupportsMultiModal):
    """Both Chatterbox stages, dispatched on ``model_stage``."""

    supports_multimodal = True
    supports_multimodal_raw_input_only = True
    requires_raw_input_tokens = True
    # Stage transport consumes codec IDs and reference features, never T3
    # hidden states. Avoid per-token host copies of an unused hidden payload.
    requires_full_prefix_cached_hidden_states = False
    omni_pooler_payload_include_hidden = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config: ChatterboxMTLV3Config = vllm_config.model_config.hf_config
        self.model_stage = vllm_config.model_config.model_stage
        self.model_dir = resolve_model_dir(vllm_config.model_config.model)
        self.have_multimodal_outputs = True

        # Per-request acoustic reference conditioning, captured at prefill and
        # replayed in batch order on every later step (see `_reference_payload`).
        self._ref_by_req: dict[str, dict[str, torch.Tensor]] = {}
        # Incremental streaming: the connector delivers codes a few at a time,
        # so the acoustic stage keeps each request's cumulative sequence and
        # its own decode schedule (see _forward_s3gen).
        self._stream_codes: dict[str, torch.Tensor] = {}
        self._stream_next_decode: dict[str, int] = {}
        self._stream_block: dict[str, int] = {}
        self._stream_cond: dict[str, ReferenceConditioning] = {}
        #: Cumulative code count at this stream's previous decode -- used to keep
        #: the bounded context window wide enough that consecutive windows always
        #: overlap the region already emitted.
        self._stream_last_decode_at: dict[str, int] = {}
        #: Bounded left-context window for the streaming re-decode, in codes.
        #: 0 (default) = re-decode the whole prefix every chunk.
        self._stream_ctx_window = int(
            getattr(self.config, "acoustic_stream_ctx_window", K.ACOUSTIC_STREAM_CTX_WINDOW)
        )
        self._batch_req_ids: list[str] = []

        if self.model_stage == K.T3_STAGE:
            self.t3 = ChatterboxT3(vllm_config=backbone_vllm_config(vllm_config), prefix=prefix)
            self.model = self.t3
            self.s3gen = None
        elif self.model_stage == K.S3GEN_STAGE:
            self.t3 = None
            self.s3gen = ChatterboxS3Gen(
                checkpoint_profile=self.config.checkpoint_profile,
                cfm_timesteps=self.config.acoustic_cfm_timesteps,
                flow_cudagraphs=getattr(self.config, "acoustic_flow_cudagraphs", False),
                compile_estimator=getattr(self.config, "acoustic_compile_estimator", False),
                estimator_dtype=getattr(self.config, "acoustic_estimator_dtype", "float32"),
                graph_cache_size=getattr(self.config, "acoustic_graph_cache_size", 16),
                graph_max_codes=getattr(self.config, "acoustic_graph_max_codes", 75),
                prewarm_prompt_tokens=getattr(self.config, "acoustic_prewarm_prompt_tokens", ()),
                materialize_vocoder_weights=getattr(self.config, "acoustic_materialize_vocoder_weights", False),
                vocoder_cudagraphs=getattr(self.config, "acoustic_vocoder_cudagraphs", False),
                max_batch_rows=getattr(
                    self.config, "acoustic_max_batch_rows", K.ACOUSTIC_MAX_BATCH_ROWS
                ),
                watermark_workers=getattr(
                    self.config, "acoustic_watermark_workers", K.ACOUSTIC_WATERMARK_WORKERS
                ),
                watermark_device=getattr(
                    self.config, "acoustic_watermark_device", K.ACOUSTIC_WATERMARK_DEVICE
                ),
                batch_vocoder=getattr(self.config, "acoustic_batch_vocoder", False),
            )
            self.model = self.s3gen
            self.enable_update_additional_information = True
        else:
            raise ValueError(f"unsupported model_stage {self.model_stage!r}")

    def get_language_model(self) -> nn.Module:
        return self.model

    def prepare_runner_inputs(self, *, input_ids, positions, req_ids, **kwargs):
        """Record this step's request ids, in persistent-batch order.

        ``forward`` gets per-request tensors positionally but no ids, and the
        downstream payload split indexes lists by **request index**. Without the
        ids there is no way to put a request's own reference conditioning at its
        own index, and the splitter's ``element[0]`` fallback then hands one
        caller's voice to another.
        """
        self._batch_req_ids = list(req_ids)
        computed = kwargs.get("num_computed_tokens")
        scheduled = kwargs.get("num_scheduled_tokens")
        self._batch_positions_cpu = None
        if self.model_stage == K.T3_STAGE and computed is not None and scheduled is not None:
            # These are the runner's CPU scheduling arrays, before any graph
            # execution. Derive the same absolute positions without a D2H copy.
            self._batch_positions_cpu = [
                p for start, count in zip(computed, scheduled)
                for p in range(int(start), int(start) + int(count))
            ]
        return input_ids, positions

    def on_requests_finished(self, finished_req_ids) -> None:
        """Drop per-request state so a finished voice cannot leak or leak memory."""
        ids = list(finished_req_ids or ())
        for req_id in ids:
            self._ref_by_req.pop(req_id, None)
            self._stream_codes.pop(req_id, None)
            self._stream_next_decode.pop(req_id, None)
            self._stream_block.pop(req_id, None)
            self._stream_cond.pop(req_id, None)
            self._stream_last_decode_at.pop(req_id, None)
        # A cancelled stream leaves a held-back crossfade tail behind; drop it
        # so a barge-in cannot retain a caller's audio.
        s3gen = getattr(self, "s3gen", None)
        if s3gen is not None and hasattr(s3gen, "release_stream_state"):
            s3gen.release_stream_state(ids)

    # ------------------------------------------------------------------
    # Stage 0: embeddings
    # ------------------------------------------------------------------
    def embed_multimodal(self, **kwargs: object):
        """Build each request's 34-position conditioning prefix.

        vLLM batches every multimodal item scheduled in one engine step into a
        single call, and a guided request contributes TWO rows (its conditional
        and its unconditional companion), so this is a batched path from the
        very first request -- it must never assume one item.

        The per-item speech prompt is sliced back to its own
        ``cond_prompt_len``: collation right-pads shorter prompts, and feeding
        the padding through the Perceiver would change the voice for anyone
        whose reference clip is under six seconds.
        """
        if self.model_stage != K.T3_STAGE:
            raise RuntimeError(f"embed_multimodal is only valid for {K.T3_STAGE}")

        speaker = kwargs["speaker_emb"]
        tokens = kwargs["cond_prompt_speech_tokens"]
        exaggeration = kwargs["exaggeration"]
        lengths = kwargs.get("cond_prompt_len")

        def one(spk, tok, exa, n_prompt):
            tok = torch.as_tensor(tok).reshape(1, -1)
            if n_prompt is not None:
                n_prompt = int(n_prompt)
                if 0 < n_prompt < tok.shape[1]:
                    tok = tok[:, :n_prompt]
            return self.t3.prepare_conditioning(
                torch.as_tensor(spk).reshape(1, -1),
                tok,
                torch.as_tensor(exa).reshape(1, 1, 1),
            ).reshape(K.COND_PREFIX_LEN, -1)

        if isinstance(speaker, (list, tuple)):
            items = list(zip(speaker, tokens, exaggeration))
            lens = list(lengths) if isinstance(lengths, (list, tuple)) else [None] * len(items)
        else:
            spk = torch.as_tensor(speaker).reshape(-1, K.SPEAKER_EMBED_SIZE)
            count = int(spk.shape[0])
            tok = torch.as_tensor(tokens).reshape(count, -1)
            exa = torch.as_tensor(exaggeration).reshape(count)
            items = [(spk[i], tok[i], exa[i]) for i in range(count)]
            if lengths is None:
                lens = [None] * count
            else:
                lens = torch.as_tensor(lengths).reshape(-1).tolist()
                lens = (lens + [None] * count)[:count]
        return [one(spk, tok, exa, n) for (spk, tok, exa), n in zip(items, lens)]

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        """Fill ONLY the conditioning positions.

        Text / BOS / decode content and both learned position tables are applied
        in :meth:`forward`, which is the first place that knows each row's
        position inside its own request. Dispatching on the token id instead
        would be wrong: a codec id and a text id are the same integers.
        """
        if self.model_stage == K.S3GEN_STAGE:
            return torch.zeros(
                (input_ids.shape[0], int(self.config.hidden_size)),
                device=input_ids.device,
                dtype=torch.float32,
            )

        dim = K.HIDDEN_SIZE
        dtype = self.t3.speech_emb.weight.dtype
        out = torch.zeros((input_ids.shape[0], dim), device=input_ids.device, dtype=dtype)
        if multimodal_embeddings is None or is_multimodal is None:
            return out

        mask = is_multimodal.to(torch.bool).reshape(-1)
        if not bool(mask.any()):
            return out
        embeds = multimodal_embeddings
        if isinstance(embeds, (list, tuple)):
            embeds = torch.cat([e.reshape(-1, dim) for e in embeds], dim=0)
        else:
            embeds = embeds.reshape(-1, dim)
        out[mask] = embeds.to(dtype=out.dtype)
        return out

    def _apply_positions_and_content(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        request_token_spans: list[tuple[int, int]] | None,
        infos: list[dict] | None,
        extra_args: list[dict] | None,
    ) -> tuple[torch.Tensor, list[int]]:
        """Apply per-request content + learned positions to non-conditioning rows.

        For a request with ``text_len == T`` and ``prompt_len == 34 + T + 2``,
        a row at global position ``p`` gets:

        ==================  ==========================================================
        ``p < 34``          conditioning (already written by ``embed_input_ids``)
        ``34 <= p < 34+T``  ``text_emb(id) + text_pos[p-34]``, content zeroed if uncond
        ``34+T <= p < L``   ``speech_emb[BOS] + speech_pos[0]``  (both BOS rows)
        ``p >= L``          ``speech_emb(id) + speech_pos[p-L+1]``
        ==================  ==========================================================

        The last line is the contract that matters most: the learned speech
        position comes from this request's own progress, never from a batch-wide
        counter and never from the global transformer position.
        """
        prefill_indices: list[int] = []
        if not request_token_spans:
            return inputs_embeds, prefill_indices

        # Use the runner's CPU positions for the whole scheduled batch. The old path
        # asked the GPU four yes/no questions PER sequence and then used
        # dynamically sized boolean gathers. At 30 CFG pairs that serialized
        # hundreds of device synchronizations on every autoregressive step.
        # Metadata is request-local; only embedding lookups run on the GPU.
        host_positions = getattr(self, "_batch_positions_cpu", None)
        if host_positions is None:
            # Standalone tests and callers outside the model runner.
            host_positions = positions.detach().cpu().tolist()
        text_rows, text_positions, text_conditional = [], [], []
        bos_rows, decode_rows, decode_positions = [], [], []
        for req_idx, (start, end) in enumerate(request_token_spans):
            if end <= start:
                continue
            info = infos[req_idx] if infos and req_idx < len(infos) else {}
            text_len = int(info.get("text_len", 0) or 0)
            prompt_len = int(info.get("prompt_len", 0) or 0)
            if prompt_len <= 0:
                raise RuntimeError(
                    "Chatterbox T3: request is missing prompt_len/text_len in its "
                    "additional_information; the adapter must supply them"
                )
            role = str(info.get("cfg_role") or "cond")
            if extra_args and req_idx < len(extra_args):
                role = str(extra_args[req_idx].get("cfg_role") or "cond")
            is_prefill = False
            for row in range(start, end):
                pos = host_positions[row]
                if pos < K.COND_PREFIX_LEN:
                    is_prefill = True
                elif pos < K.COND_PREFIX_LEN + text_len:
                    text_rows.append(row)
                    text_positions.append(pos - K.COND_PREFIX_LEN)
                    text_conditional.append(role != "uncond")
                elif pos < prompt_len:
                    bos_rows.append(row)
                else:
                    decode_rows.append(row)
                    decode_positions.append(pos - prompt_len + 1)
            if is_prefill:
                prefill_indices.append(req_idx)

        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        def indices(values):
            return torch.tensor(values, dtype=torch.long, device=device)

        if text_rows:
            rows = indices(text_rows)
            content = self.t3.text_content_embedding(input_ids[rows].long()).to(dtype)
            conditional = torch.tensor(text_conditional, device=device, dtype=torch.bool)
            content = torch.where(conditional[:, None], content, 0.0)
            inputs_embeds[rows] = content + self.t3.text_position_embedding(indices(text_positions)).to(dtype)
        if bos_rows:
            inputs_embeds[indices(bos_rows)] = self.t3.bos_embedding().to(dtype)
        if decode_rows:
            rows = indices(decode_rows)
            inputs_embeds[rows] = (
                self.t3.speech_content_embedding(input_ids[rows].long())
                + self.t3.speech_position_embedding(indices(decode_positions))
            ).to(dtype)
        return inputs_embeds, prefill_indices

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> OmniOutput:
        if self.model_stage == K.T3_STAGE:
            return self._forward_t3(input_ids, positions, inputs_embeds, **kwargs)
        return self._forward_s3gen(input_ids, **kwargs)

    def _forward_t3(self, input_ids, positions, inputs_embeds, **kwargs) -> OmniOutput:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        infos = kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information")
        spans = kwargs.get("request_token_spans")
        extra_args = kwargs.get("sampling_extra_args")
        inputs_embeds, prefill_indices = self._apply_positions_and_content(
            inputs_embeds,
            input_ids,
            positions,
            list(spans) if spans else None,
            list(infos) if isinstance(infos, list) else None,
            list(extra_args) if isinstance(extra_args, list) else None,
        )

        hidden_states = self.t3(inputs_embeds, positions)

        multimodal_outputs = self._reference_payload(kwargs, prefill_indices)
        return OmniOutput(text_hidden_states=hidden_states, multimodal_outputs=multimodal_outputs)

    def _reference_payload(self, kwargs: dict, prefill_indices: list[int]) -> dict[str, Any]:
        """Per-request acoustic reference conditioning, in batch order.

        The downstream splitter indexes these lists by **request index** and
        falls back to ``element[0]`` when the list is shorter than the batch.
        Multimodal kwargs, however, only arrive for the requests that are
        PREFILLING this step, and vLLM packs decode requests first -- so
        emitting the mm items in their own order attributes one caller's voice
        to whoever happens to occupy that index. That is a voice crossover, not
        a cosmetic mismatch.

        Instead: capture each request's conditioning at its prefill step, keyed
        by request id, and emit a full batch-length list every step.
        """
        req_ids = list(self._batch_req_ids)
        if not req_ids:
            return {}

        if "speech_token" in kwargs:
            tokens = _per_request(kwargs.get("speech_token"), 2) or []
            lens = _per_request(kwargs.get("speech_token_len"), 2) or []
            feats = _per_request(kwargs.get("speech_feat"), 3) or []
            embs = _per_request(kwargs.get("embedding"), 2) or []
            if len(tokens) != len(prefill_indices):
                logger.warning_once(
                    "Chatterbox T3: %d multimodal reference items for %d prefilling requests; "
                    "refusing to guess the mapping",
                    len(tokens), len(prefill_indices),
                )
            else:
                for slot, idx in enumerate(prefill_indices):
                    if idx >= len(req_ids):
                        continue
                    reference = {
                        "speech_token": tokens[slot],
                        "speech_token_len": lens[slot] if slot < len(lens) else None,
                        "speech_feat": feats[slot] if slot < len(feats) else None,
                        "embedding": embs[slot] if slot < len(embs) else None,
                    }
                    # These features are constant for the entire utterance.
                    # Stage transport is CPU-based; materialize once instead
                    # of recopying each caller's reference on every AR token.
                    self._ref_by_req[req_ids[idx]] = {
                        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                        for key, value in reference.items()
                    }

        known = [self._ref_by_req.get(rid) for rid in req_ids]
        if not any(k is not None for k in known):
            return {}

        def column(key: str) -> list:
            # A request with no conditioning yet gets an EMPTY tensor, never
            # another request's: the acoustic stage treats empty as "missing"
            # and refuses to synthesise rather than substituting a voice.
            out = []
            for entry in known:
                value = entry.get(key) if entry else None
                out.append(value if value is not None else torch.zeros(0))
            return out

        return to_dict(
            OmniPayloadStruct(
                embed=EmbeddingsStruct(
                    speech_token=column("speech_token"),
                    speech_token_len=column("speech_token_len"),
                    speech_feat=column("speech_feat"),
                    embedding=column("embedding"),
                ),
            )
        )

    def _forward_s3gen(self, input_ids, **kwargs) -> OmniOutput:
        runtime = kwargs.get("model_intermediate_buffer")
        if runtime is None:
            runtime = kwargs.get("runtime_additional_information", [])
        if not isinstance(runtime, list):
            runtime = []

        seq_token_counts = kwargs.get("seq_token_counts")
        flat = input_ids.reshape(-1).to(torch.long)
        per_request = _split_ids(flat, seq_token_counts)

        sample_rate = torch.tensor(int(self.config.sample_rate), dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32, device=input_ids.device)
        n = max(1, len(per_request))
        audios: list[torch.Tensor] = [empty] * n
        srs = [sample_rate] * n

        jobs: list[AcousticRequest] = []
        job_rows: list[int] = []
        for idx, codes in enumerate(per_request):
            raw = runtime[idx] if idx < len(runtime) and isinstance(runtime[idx], dict) else {}
            # The scheduler attaches this transport-only flag when a stream
            # is resumed. It is not part of the model's MetaStruct schema.
            if isinstance(raw.get("meta"), dict) and "resumable" in raw["meta"]:
                raw = {**raw, "meta": {k: v for k, v in raw["meta"].items() if k != "resumable"}}
            payload = to_struct(raw)
            embed = payload.embed
            meta = payload.meta
            # Identity and terminal state are read BEFORE the payload guards.
            # The last event of a stream is a marker: it carries no codes and no
            # reference, because everything was already sent. Skipping it on
            # those grounds would leave the buffered tail of every utterance
            # undecoded -- the clause would simply stop mid-sentence.
            req_id = (meta.req_id[0] if (meta and meta.req_id) else None) or f"row{idx}"
            chunked = bool(meta is not None and meta.chunk_seq is not None)
            is_final = True
            if chunked and meta is not None and meta.stream_finished is not None:
                is_final = bool(meta.stream_finished.reshape(-1)[0].item())

            if _STREAM_DEBUG:
                logger.info(
                    "[cbx-row] idx=%d req=%s chunked=%s chunk_seq=%s "
                    "stream_finished=%s finished=%s codes_in=%d",
                    idx, req_id, chunked,
                    None if meta is None else meta.chunk_seq,
                    None if (meta is None or meta.stream_finished is None)
                    else bool(meta.stream_finished.reshape(-1)[0].item()),
                    None if (meta is None or meta.finished is None)
                    else bool(meta.finished.reshape(-1)[0].item()),
                    int(codes.numel()),
                )

            has_reference = not (
                embed is None
                or embed.speech_token is None
                or embed.speech_feat is None
                or embed.speech_token.numel() == 0
                or embed.speech_feat.numel() == 0
            )
            valid = codes[(codes >= 0) & (codes < K.CODEC_VOCAB_SIZE)]
            # Acoustic noise runs on a FIXED, reproducible stream by default, so
            # the same codec sequence and reference always decode to the same
            # waveform. The request's `seed` controls AR sampling -- which is
            # what decides the words and prosody; the acoustic stage's noise is
            # deliberately not made request-unique, because that would make
            # `seed` fail to reproduce a result.
            seed = int(meta.audio_seed.item()) if (meta and meta.audio_seed is not None) else 0
            cond = self._stream_cond.get(req_id) if chunked else None
            if cond is None and has_reference:
                try:
                    cond = self._reference_from_payload(embed, input_ids.device, req_id)
                except AcousticPayloadError as exc:
                    logger.error("Chatterbox S3Gen: %s", exc)
                    continue
                if chunked:
                    self._stream_cond[req_id] = cond
            elif cond is None and chunked:
                # Terminal marker: reuse the voice this stream has been using.
                cond = self._stream_cond.get(req_id)
            if cond is None:
                if valid.numel():
                    logger.warning_once(
                        "Chatterbox S3Gen: %d codec tokens arrived without reference "
                        "conditioning; refusing to substitute a default voice",
                        int(valid.numel()),
                    )
                continue
            # In chunked mode the connector drives this stage like an AR
            # decoder: each forward carries only the codes produced since the
            # last one (often a single token), NOT the sequence so far. The
            # flow decoder needs the whole prefix -- it is conditioned on a
            # bidirectional token encoder, so a lone code has no context and
            # renders as a fragment. The cumulative buffer therefore lives
            # here, and this stage also decides WHEN a decode is worth doing.
            if not chunked:
                if valid.numel() == 0:
                    continue
                jobs.append(AcousticRequest(req_id, valid, cond, seed=seed, finalize=True))
                job_rows.append(idx)
                continue

            prefix = self._stream_codes.get(req_id)
            if valid.numel():
                prefix = valid if prefix is None else torch.cat([prefix, valid])
                self._stream_codes[req_id] = prefix
            if prefix is None or prefix.numel() == 0:
                continue
            n_codes = int(prefix.numel())

            if req_id not in self._stream_next_decode:
                self._stream_next_decode[req_id] = K.ACOUSTIC_STREAM_FIRST_BLOCK
                self._stream_block[req_id] = K.ACOUSTIC_STREAM_FIRST_BLOCK

            scheduled = self._stream_next_decode[req_id]
            if not is_final and n_codes < scheduled:
                continue  # not enough new audio to be worth a decode yet

            # Decode EXACTLY the scheduled length, not everything accumulated.
            # Acoustic rows may only share a forward pass when their code counts
            # match (padding is not isolated by this checkpoint's token encoder,
            # measured at 40% waveform corruption). Left to drift, every
            # concurrent stream sits at a different count and NOTHING batches --
            # each decode is a batch of one, serialised at ~100 ms, which is
            # what builds the queue under load. Pinning every stream to the same
            # ladder (5, 15, 35, 75, ...) makes concurrent first chunks
            # length-identical, so they batch. Mixed voices at equal length are
            # already gated safe (0.008-0.017% of peak), so this costs nothing
            # in quality; the codes past the scheduled point are not dropped,
            # they arrive in the next chunk (the prefix is cumulative).
            # Bounded left-context window: once the utterance is longer than the
            # configured window, a non-final chunk re-decodes only its recent
            # tail (still prepended with the full reference prompt) instead of
            # the whole prefix, so per-chunk acoustic cost stops growing and
            # equal-window rows re-batch. The window is widened when needed so it
            # always covers the previous decode point (no emit gap). is_final
            # always decodes the whole prefix (token_offset 0) -- the final crop,
            # watermark and one-shot parity are unchanged.
            window = 0
            if not is_final and self._stream_ctx_window > 0 and scheduled > self._stream_ctx_window:
                last_at = self._stream_last_decode_at.get(req_id, 0)
                w_eff = max(self._stream_ctx_window, scheduled - last_at + 96)
                window = max(0, scheduled - w_eff)
            decode_codes = prefix if is_final else prefix[window:scheduled]
            if not is_final:
                self._stream_last_decode_at[req_id] = scheduled
                block = min(
                    int(self._stream_block[req_id] * K.ACOUSTIC_STREAM_BLOCK_GROWTH),
                    K.ACOUSTIC_STREAM_MAX_BLOCK,
                )
                self._stream_block[req_id] = block
                self._stream_next_decode[req_id] = scheduled + block

            jobs.append(
                AcousticRequest(
                    req_id, decode_codes, cond, seed=seed, finalize=is_final,
                    streaming=True, token_offset=window,
                )
            )
            job_rows.append(idx)
            if is_final:
                self._stream_codes.pop(req_id, None)
                self._stream_next_decode.pop(req_id, None)
                self._stream_block.pop(req_id, None)
                self._stream_cond.pop(req_id, None)
                self._stream_last_decode_at.pop(req_id, None)

        if jobs:
            for row, result in zip(job_rows, self.s3gen.decode(jobs)):
                audios[row] = result.audio.to(torch.float32)
                if _STREAM_DEBUG:
                    job = jobs[job_rows.index(row)]
                    c = job.conditioning
                    logger.info(
                        "[cbx-stream] req=%s codes=%d first=%s finalize=%s "
                        "emitted=%d (%.3fs) | prompt_len=%s prompt_tok_sum=%.1f "
                        "feat=%s emb_norm=%.4f",
                        job.request_id, int(job.codes.numel()),
                        job.codes[:5].tolist(), job.finalize,
                        int(result.audio.numel()), float(result.audio.numel()) / 24000.0,
                        int(c.prompt_token_len.reshape(-1)[0].item()),
                        float(c.prompt_token.float().sum().item()),
                        tuple(c.prompt_feat.shape),
                        float(c.embedding.float().norm().item()),
                    )

        return OmniOutput(text_hidden_states=None, multimodal_outputs={"audio": audios, "sr": srs})


    def _reference_from_payload(
        self, embed, device: torch.device, req_id: str
    ) -> ReferenceConditioning:
        """Rebuild the acoustic reference from a transferred payload.

        Shapes are taken from the reference's OWN declared
        ``prompt_token_len``, not from whatever shape the transport happened to
        deliver. A payload that carries more than one row (e.g. a collated
        batch) is sliced back to one reference rather than flattened -- a
        flatten would silently double the reference length and produce a
        different voice.
        """
        spk_dim = int(self.s3gen.s3gen.flow.spk_embed_affine_layer.in_features)
        mel_dim = int(self.s3gen.s3gen.flow.output_size)

        token = embed.speech_token
        feat = embed.speech_feat
        emb = embed.embedding
        if token is None or feat is None or emb is None:
            raise AcousticPayloadError(f"{req_id}: incomplete acoustic reference payload")

        token = token.reshape(-1).to(device=device, dtype=torch.long)
        emb = emb.reshape(-1, spk_dim).to(device)
        feat = feat.reshape(-1, mel_dim).to(device)

        declared = None
        if embed.speech_token_len is not None:
            values = embed.speech_token_len.reshape(-1)
            if values.numel():
                declared = int(values[0].item())
        if declared is None or declared <= 0 or declared > token.numel():
            declared = int(token.numel())

        n_frames = declared * int(self.config.token_mel_ratio)
        if feat.shape[0] < n_frames:
            raise AcousticPayloadError(
                f"{req_id}: reference has {feat.shape[0]} mel frames but "
                f"{declared} codes need {n_frames}"
            )
        if emb.shape[0] > 1 or token.numel() > declared or feat.shape[0] > n_frames:
            logger.warning_once(
                "Chatterbox S3Gen: reference payload carried more rows than one request "
                "needs (embedding %s, tokens %d vs %d, mel %d vs %d); slicing to the "
                "declared reference length",
                tuple(emb.shape), int(token.numel()), declared, int(feat.shape[0]), n_frames,
            )

        return ReferenceConditioning(
            cache_key=req_id,
            speaker_emb=torch.zeros(1, K.SPEAKER_EMBED_SIZE, device=device),
            cond_prompt_speech_tokens=torch.zeros(1, 1, dtype=torch.long, device=device),
            prompt_token=token[:declared].reshape(1, -1),
            prompt_token_len=torch.tensor([declared], dtype=torch.long, device=device),
            prompt_feat=feat[:n_frames].reshape(1, n_frames, mel_dim),
            embedding=emb[:1],
            checkpoint_profile=self.config.checkpoint_profile,
            prompt_token_count=declared,
        )

    # ------------------------------------------------------------------
    def compute_logits(self, hidden_states) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if self.model_stage != K.T3_STAGE:
            raise RuntimeError(f"compute_logits is only valid for {K.T3_STAGE}")
        return self.t3.compute_speech_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from safetensors.torch import load_file as load_safetensors

        if self.model_stage == K.T3_STAGE:
            path = os.path.join(self.model_dir, self.config.t3_checkpoint)
            state = load_safetensors(path)
            loaded = self.t3.load_weights(state.items())
            logger.info(
                "Chatterbox MTL V3: loaded T3 %s (profile %s), %d parameters",
                self.config.t3_checkpoint, self.config.checkpoint_profile, len(loaded),
            )
            # The loader tracks coverage against THIS module's parameter names,
            # and ChatterboxT3 is installed under `t3.`; returning its own
            # relative names would report every tensor as uninitialised.
            return {f"t3.{name}" for name in loaded}
        self.s3gen.load_weights_from_dir(self.model_dir)
        self.s3gen.warm_compiled_estimator()
        return {f"s3gen.{name}" for name, _ in self.s3gen.named_parameters()}


def _per_request(value, want_dim: int) -> list[torch.Tensor] | None:
    """Split a collated multimodal batch tensor into one tensor per request.

    Must not trigger a host<->device sync: it runs inside the AR forward, which
    may be CUDA-graph captured.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = list(value)
    elif torch.is_tensor(value):
        items = [value[i] for i in range(value.shape[0])]
    else:
        return None
    out = []
    for t in items:
        if torch.is_tensor(t):
            while t.dim() < want_dim:
                t = t.unsqueeze(0)
            if t.shape[0] != 1:
                t = t[:1]
            t = t.contiguous()
        out.append(t)
    return out


def _split_ids(flat: torch.Tensor, seq_token_counts) -> list[torch.Tensor]:
    if seq_token_counts:
        counts = [int(c) for c in seq_token_counts]
        out, off = [], 0
        for c in counts:
            out.append(flat[off : off + c])
            off += c
        return out
    return [flat]
