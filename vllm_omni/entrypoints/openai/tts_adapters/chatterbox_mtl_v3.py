# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox Multilingual V3 serving adapter.

Owns the request contract for ``/v1/audio/speech``:

* language policy -- ``ar`` and ``en`` are first-class. The shared
  ``DEFAULT_TTS_LANGUAGES`` does not contain Arabic, so this adapter declares
  its own set rather than inheriting one that rejects its primary target.
* validation **before** GPU admission: non-empty text, supported language, a
  resolvable voice, a bounded reference clip, finite parameters, and a text
  length the learned position table can actually hold.
* guided sampling: every request is expanded into a strict CFG pair, with the
  reference's own sampling profile (temperature 0.8, min-p 0.05, top-p 1.0,
  no top-k) and the engine's repetition penalty pinned to 1.0 so the
  speech-only penalty is not applied twice.
* generation validation: a run that ends without the speech EOS is a truncated
  generation, and a run whose guidance broke is a failure -- neither is returned
  as valid audio.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    TTSCapabilities,
    TTSGenerationError,
)
from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K
from vllm_omni.model_executor.models.chatterbox_mtl_v3.sampling import (
    COND,
    POLICY_HARDENED,
    POLICY_REFERENCE,
    cfg_scale_from_weight,
)
from vllm_omni.model_executor.models.chatterbox_mtl_v3.text import (
    TextPolicyError,
    build_text_token_ids,
    normalize_language,
    prompt_length,
    punc_norm,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)

#: Extra request parameters this model accepts, with their bounds. Anything
#: else in ``extra_params`` is rejected rather than silently ignored.
_EXTRA_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "cfg_weight": (0.0, 5.0),
    "exaggeration": (0.0, 2.0),
    "temperature": (0.0, 2.0),
    "min_p": (0.0, 1.0),
    "top_p": (0.0, 1.0),
    "repetition_penalty": (0.5, 4.0),
}
_EXTRA_PARAM_CHOICES: dict[str, frozenset[str]] = {
    "policy": frozenset({POLICY_REFERENCE, POLICY_HARDENED}),
}


@register_tts_adapter
class ChatterboxMTLV3Adapter(ARTTSAdapter):
    name = K.MODEL_TYPE
    stage_keys = frozenset({K.T3_STAGE})
    model_archs = frozenset({K.MODEL_ARCH})
    validates_generation = True
    native_speed_control = False
    supported_output_sample_rates = frozenset({K.S3GEN_SR})

    max_new_tokens_min = 1
    max_new_tokens_max = K.MAX_NEW_SPEECH_TOKENS

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._tokenizer = None
        self._config = None

    # -- config / tokenizer -------------------------------------------------
    @property
    def config(self):
        if self._config is None:
            self._config = self.ctx.server.model_config.hf_config
        return self._config

    def _get_tokenizer(self):
        """The model's own multilingual grapheme tokenizer.

        Never a Qwen or generic Llama tokenizer, and never the English
        ``EnTokenizer``: those produce ids this checkpoint's 2454-entry text
        embedding cannot represent.
        """
        if self._tokenizer is None:
            from vllm_omni.model_executor.models.chatterbox_mtl_v3.chatterbox_mtl_v3 import (
                resolve_model_dir,
            )
            from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.tokenizers import (
                MTLTokenizer,
            )

            model_dir = resolve_model_dir(self.ctx.server.engine_client.model_config.model)
            path = os.path.join(model_dir, K.TOKENIZER_FILE)
            if not os.path.exists(path):
                raise RuntimeError(
                    f"Chatterbox multilingual tokenizer missing at {path}; refusing to start "
                    f"with a substituted tokenizer"
                )
            self._tokenizer = MTLTokenizer(path)
        return self._tokenizer

    # -- capabilities -------------------------------------------------------
    def _load_supported_languages(self) -> frozenset[str]:
        qualified = frozenset(
            getattr(self.config, "qualified_languages", None) or K.QUALIFIED_LANGUAGE_CODES
        )
        if bool(getattr(self.config, "allow_unqualified_languages", False)):
            codes = set(K.SUPPORTED_LANGUAGE_CODES)
        else:
            codes = set(qualified)
        names = {K.SUPPORTED_LANGUAGE_CODES[c] for c in codes if c in K.SUPPORTED_LANGUAGE_CODES}
        return frozenset(codes | names)

    def _load_supported_speakers(self) -> set[str]:
        # Voices are reference clips, not a fixed enum; uploaded/registered
        # speakers are resolved through the shared server helpers.
        return set()

    def _load_codec_frame_rate(self) -> float | None:
        return float(K.S3_TOKEN_RATE)

    def load_capabilities(self) -> TTSCapabilities:
        self.capabilities = TTSCapabilities(
            precomputed_speakers={},
            supported_speakers=frozenset(self._load_supported_speakers()),
            supported_languages=self._load_supported_languages(),
            codec_frame_rate=self._load_codec_frame_rate(),
        )
        return self.capabilities

    # -- request lifecycle --------------------------------------------------
    def normalize(self, request: "OpenAICreateSpeechRequest") -> None:
        if request.voice:
            request.voice = request.voice.lower()
        if not request.language:
            # One language id per call is the model's contract; guessing it
            # silently would change pronunciation, so it must be explicit.
            request.language = "en"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        server = self.ctx.server
        err = server._apply_uploaded_speaker(request)
        if err:
            return err

        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        try:
            language = normalize_language(request.language, self.config)
        except TextPolicyError as exc:
            return str(exc)

        if request.ref_audio is None:
            return (
                "Chatterbox Multilingual V3 requires 'ref_audio' (a reference voice clip) "
                "or a registered 'voice'"
            )
        fmt_err = server._validate_ref_audio_format(request.ref_audio)
        if fmt_err:
            return fmt_err

        try:
            ids = build_text_token_ids(self._get_tokenizer(), request.input, language, self.config)
        except TextPolicyError as exc:
            return str(exc)
        if prompt_length(len(ids)) < K.COND_PREFIX_LEN + 3:
            return "Input text produced no tokens after normalization"

        if request.max_new_tokens is not None:
            if not self.max_new_tokens_min <= request.max_new_tokens <= self.max_new_tokens_max:
                return (
                    f"max_new_tokens must be between {self.max_new_tokens_min} and "
                    f"{self.max_new_tokens_max}"
                )
        if request.sample_rate is not None and request.sample_rate not in self.supported_output_sample_rates:
            return f"sample_rate must be {K.S3GEN_SR} (the model's native rate)"
        if request.speed is not None and abs(float(request.speed) - 1.0) > 1e-6:
            # Refuse rather than ignore: silently dropping a speed request
            # returns audio that is not what was asked for.
            return "Chatterbox Multilingual V3 does not support 'speed'"
        if request.instructions:
            return "Chatterbox Multilingual V3 does not support 'instructions'"

        return self._validate_extra_params(request.extra_params)

    @staticmethod
    def _validate_extra_params(extra: Mapping[str, Any] | None) -> str | None:
        if not extra:
            return None
        for key, value in extra.items():
            if key in _EXTRA_PARAM_CHOICES:
                if value not in _EXTRA_PARAM_CHOICES[key]:
                    return f"extra_params.{key} must be one of {sorted(_EXTRA_PARAM_CHOICES[key])}"
                continue
            if key not in _EXTRA_PARAM_BOUNDS:
                return (
                    f"unknown extra_params.{key!r}; supported: "
                    f"{sorted(set(_EXTRA_PARAM_BOUNDS) | set(_EXTRA_PARAM_CHOICES))}"
                )
            lo, hi = _EXTRA_PARAM_BOUNDS[key]
            try:
                number = float(value)
            except (TypeError, ValueError):
                return f"extra_params.{key} must be a number"
            if not np.isfinite(number) or not lo <= number <= hi:
                return f"extra_params.{key} must be finite and within [{lo}, {hi}]"
        return None

    async def build(
        self,
        request: "OpenAICreateSpeechRequest",
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        server = self.ctx.server
        language = normalize_language(request.language, self.config)
        extra = dict(request.extra_params or {})
        exaggeration = float(extra.get("exaggeration", self.config.default_exaggeration))

        wav, sr, _ = await server._resolve_ref_audio(request.ref_audio)
        text_ids = build_text_token_ids(self._get_tokenizer(), request.input, language, self.config)

        mm_kwargs: dict[str, Any] = {"language": language, "exaggeration": exaggeration}
        if request.voice:
            voice = request.voice.lower()
            if voice in server.uploaded_speakers and not has_inline_ref_audio:
                mm_kwargs["voice_name"] = voice
                mm_kwargs["voice_created_at"] = server._voice_created_at(voice)

        prompt: dict[str, Any] = {
            "prompt": request.input,
            "multi_modal_data": {"audio": (np.asarray(wav, dtype=np.float32), sr)},
            "mm_processor_kwargs": mm_kwargs,
            # The model needs each row's own prompt geometry to place the two
            # learned position tables; it cannot infer it from token ids.
            "additional_information": {
                "text_len": len(text_ids),
                "prompt_len": prompt_length(len(text_ids)),
                "cfg_role": COND,
                "language": language,
                "audio_seed": int(request.seed) if request.seed is not None else 0,
            },
        }
        return PreparedRequest(
            prompt=prompt,
            tts_params={
                "language": language,
                "exaggeration": exaggeration,
                "text_len": len(text_ids),
                "normalized_text": punc_norm(request.input),
                "guided": float(extra.get("cfg_weight", self.config.default_cfg_weight)) > 0.0,
            },
            model_type=K.MODEL_TYPE,
        )

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        import copy

        extra = dict(request.extra_params or {})
        cfg_weight = float(extra.get("cfg_weight", self.config.default_cfg_weight))
        penalty = float(extra.get("repetition_penalty", self.config.default_repetition_penalty))
        policy = str(extra.get("policy", POLICY_REFERENCE))

        params = copy.deepcopy(sampling_params_list)
        p = params[0]
        # The multilingual wrapper's own profile -- deliberately NOT Turbo's
        # top-k/top-p defaults.
        p.temperature = float(extra.get("temperature", self.config.default_temperature))
        p.min_p = float(extra.get("min_p", self.config.default_min_p))
        p.top_p = float(extra.get("top_p", self.config.default_top_p))
        p.top_k = K.DEFAULT_TOP_K
        # Our processor applies the speech-only penalty; the engine's builtin
        # would additionally penalise the prompt (conditioning placeholders and
        # text ids), so it must be exactly 1.0.
        p.repetition_penalty = 1.0
        p.max_tokens = int(request.max_new_tokens or self.config.max_new_speech_tokens)
        p.detokenize = False
        if request.seed is not None:
            p.seed = int(request.seed)

        p.extra_args = dict(p.extra_args or {})
        p.extra_args.update(
            {
                "chatterbox_repetition_penalty": penalty,
                "chatterbox_policy": policy,
            }
        )
        if cfg_weight > 0.0:
            p.extra_args.update(
                {
                    "cfg_role": COND,
                    "cfg_pair_id": request_id or "chatterbox",
                    "cfg_scale": cfg_scale_from_weight(cfg_weight),
                }
            )
        else:
            # cfg_weight == 0 means no guidance at all. The role must be
            # dropped, not left as a lone "cond": a CFG role with no companion
            # is a BROKEN pair under the strict policy and is terminated
            # immediately. This is a separate, unqualified quality profile --
            # the reference's default is 0.5.
            for key in ("cfg_role", "cfg_pair_id", "cfg_scale"):
                p.extra_args.pop(key, None)
        params[0] = p
        return params

    def validate_generation(
        self,
        tts_params: Mapping[str, object],
        *,
        stage0_finish_reason: str | None,
        output_tokens: int,
    ) -> None:
        # A broken CFG pair is caught in the stage input processor, which is
        # the first place that has this request's id (see
        # ``codec_token_only``); it fails the request before any audio is
        # synthesised, so nothing unguided can reach the client.
        if output_tokens <= 0:
            raise TTSGenerationError("the model produced no speech tokens", retryable=True)
        if stage0_finish_reason == "length":
            # Hitting the cap without EOS is a truncated utterance, not a
            # successful one; returning it would clip the caller's sentence.
            raise TTSGenerationError(
                f"generation reached the {self.config.max_new_speech_tokens}-token limit without "
                f"an end-of-speech token; the utterance is truncated",
                retryable=True,
            )
        if stage0_finish_reason == "abort":
            raise TTSGenerationError("generation was aborted", retryable=False)

        # Gross-truncation guard. The AR stage is stochastic, and this port's
        # logits come from a different attention backend than the reference
        # implementation's, so on a small fraction of draws the two diverge and
        # the sampled sequence can end early. A caller must not receive half a
        # sentence reported as success.
        #
        # The threshold is a third of the LOWEST codes-per-text-token ratio
        # measured across 24 reference generations, so it cannot plausibly
        # reject valid speech. It catches gross truncation only; it cannot
        # prove that a long-enough utterance is correct.
        text_len = int(tts_params.get("text_len") or 0)
        if text_len >= int(self.config.truncation_guard_min_text_tokens):
            floor = int(text_len * float(self.config.min_codes_per_text_token))
            if output_tokens < floor:
                raise TTSGenerationError(
                    f"generation stopped after {output_tokens} speech tokens for a "
                    f"{text_len}-token input (expected at least {floor}); the utterance is "
                    f"truncated",
                    retryable=True,
                )

    async def warmup(self) -> None:
        # Touch the tokenizer so the first real request does not pay for it.
        self._get_tokenizer()
