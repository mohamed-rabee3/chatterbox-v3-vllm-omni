# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Config for Chatterbox Multilingual V3 (T3 + S3Gen).

The upstream checkpoint ships no ``config.json``: the architecture lives in
Python (``T3Config.multilingual()`` + ``LLAMA_520M_CONFIG_DICT``). This config
makes those values explicit and serializable so a deployment records exactly
which architecture, checkpoint profile and preprocessing revision it ran.

Two vocabularies coexist and must not be conflated:

* ``text_vocab_size`` (2454) — the multilingual grapheme tokenizer;
* ``vocab_size`` (8194) — the speech head's output width, and the id space the
  engine samples in. The backbone's own 8-entry embedding table is an unused
  placeholder (``backbone_placeholder_vocab_size``) and must never see a codec id.
"""

from __future__ import annotations

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K


class ChatterboxMTLV3Config(PretrainedConfig):
    model_type = K.MODEL_TYPE

    def __init__(
        self,
        checkpoint_profile: str = K.DEFAULT_CHECKPOINT_PROFILE,
        verify_artifact_hashes: bool = True,
        enable_incremental_acoustics: bool = False,
        acoustic_cfm_timesteps: int = K.DEFAULT_CFM_TIMESTEPS,
        acoustic_max_batch_rows: int = K.ACOUSTIC_MAX_BATCH_ROWS,
        qualified_languages: tuple[str, ...] = tuple(sorted(K.QUALIFIED_LANGUAGE_CODES)),
        allow_unqualified_languages: bool = False,
        max_reference_seconds: float = 30.0,
        conditioning_cache_size: int = 256,
        **kwargs,
    ) -> None:
        # vLLM stops stage 0 on the speech EOS.
        kwargs.setdefault("eos_token_id", K.STOP_SPEECH_TOKEN)
        kwargs.setdefault("architectures", [K.MODEL_ARCH])
        super().__init__(**kwargs)

        if checkpoint_profile not in K.CHECKPOINT_PROFILES:
            raise ValueError(
                f"unknown checkpoint_profile {checkpoint_profile!r}; "
                f"expected one of {sorted(K.CHECKPOINT_PROFILES)}"
            )
        self.checkpoint_profile = checkpoint_profile
        self.t3_checkpoint = K.CHECKPOINT_PROFILES[checkpoint_profile]["t3"]
        self.s3gen_checkpoint = K.CHECKPOINT_PROFILES[checkpoint_profile]["s3gen"]
        self.verify_artifact_hashes = bool(verify_artifact_hashes)
        self.artifact_sha256 = dict(K.ARTIFACT_SHA256)
        self.repo_id = K.CHATTERBOX_REPO_ID
        self.revision = K.CHATTERBOX_REVISION
        self.upstream_source_revision = K.UPSTREAM_SOURCE_REVISION
        self.preprocessor_version = K.PREPROCESSOR_VERSION
        self.tokenizer_file = K.TOKENIZER_FILE
        self.voice_encoder_file = K.VOICE_ENCODER_FILE

        # T3 backbone
        self.hidden_size = K.HIDDEN_SIZE
        self.num_hidden_layers = K.NUM_LAYERS
        self.num_attention_heads = K.NUM_HEADS
        self.num_key_value_heads = K.NUM_KV_HEADS
        self.head_dim = K.HEAD_DIM
        self.intermediate_size = K.INTERMEDIATE_SIZE
        self.rms_norm_eps = K.RMS_NORM_EPS
        # Named `backbone_*` on purpose: assigning `rope_scaling` on a
        # PretrainedConfig triggers Transformers' RoPE validator, which expects
        # a *Llama* config. This outer config describes the whole Chatterbox
        # model, not the backbone; the backbone's own LlamaConfig is built in
        # `t3.backbone_vllm_config()` from these same constants.
        self.backbone_rope_theta = K.ROPE_THETA
        self.backbone_rope_scaling = dict(K.ROPE_SCALING)
        self.hidden_act = "silu"
        self.attention_bias = False
        self.mlp_bias = False
        self.tie_word_embeddings = False
        self.backbone_max_position_embeddings = K.BACKBONE_MAX_POSITION_EMBEDDINGS
        self.backbone_placeholder_vocab_size = K.BACKBONE_PLACEHOLDER_VOCAB_SIZE

        # Vocabularies / special tokens
        self.vocab_size = K.SPEECH_VOCAB_SIZE
        self.text_vocab_size = K.TEXT_VOCAB_SIZE
        self.start_text_token = K.START_TEXT_TOKEN
        self.stop_text_token = K.STOP_TEXT_TOKEN
        self.start_speech_token = K.START_SPEECH_TOKEN
        self.stop_speech_token = K.STOP_SPEECH_TOKEN
        self.codec_vocab_size = K.CODEC_VOCAB_SIZE
        self.max_text_tokens = K.MAX_TEXT_TOKENS
        self.max_speech_tokens = K.MAX_SPEECH_TOKENS
        self.text_pos_table_size = K.TEXT_POS_TABLE_SIZE
        self.speech_pos_table_size = K.SPEECH_POS_TABLE_SIZE
        self.max_new_speech_tokens = K.MAX_NEW_SPEECH_TOKENS
        self.min_codes_per_text_token = K.MIN_CODES_PER_TEXT_TOKEN
        self.truncation_guard_min_text_tokens = K.TRUNCATION_GUARD_MIN_TEXT_TOKENS

        # Conditioning / prefill layout
        self.speaker_embed_size = K.SPEAKER_EMBED_SIZE
        self.speech_cond_prompt_len = K.SPEECH_COND_PROMPT_LEN
        self.perceiver_output_len = K.PERCEIVER_OUTPUT_LEN
        self.cond_prefix_len = K.COND_PREFIX_LEN
        self.num_prefill_bos = K.NUM_PREFILL_BOS

        # Audio
        self.sample_rate = K.S3GEN_SR
        self.s3_sample_rate = K.S3_SR
        self.token_frame_rate = K.S3_TOKEN_RATE
        self.samples_per_codec_token = K.SAMPLES_PER_CODEC_TOKEN
        self.token_mel_ratio = K.MEL_FRAMES_PER_CODEC_TOKEN
        self.acoustic_pre_lookahead_len = K.ACOUSTIC_PRE_LOOKAHEAD_LEN
        self.acoustic_cfm_timesteps = int(acoustic_cfm_timesteps)
        self.acoustic_max_batch_rows = int(acoustic_max_batch_rows)
        self.acoustic_inference_cfg_rate = K.ACOUSTIC_INFERENCE_CFG_RATE
        self.enc_cond_seconds = K.ENC_COND_SECONDS
        self.dec_cond_seconds = K.DEC_COND_SECONDS
        self.max_reference_seconds = float(max_reference_seconds)

        # Sampling defaults
        self.default_temperature = K.DEFAULT_TEMPERATURE
        self.default_repetition_penalty = K.DEFAULT_REPETITION_PENALTY
        self.default_min_p = K.DEFAULT_MIN_P
        self.default_top_p = K.DEFAULT_TOP_P
        self.default_cfg_weight = K.DEFAULT_CFG_WEIGHT
        self.default_exaggeration = K.DEFAULT_EXAGGERATION

        # Serving policy
        self.qualified_languages = tuple(qualified_languages)
        self.allow_unqualified_languages = bool(allow_unqualified_languages)
        self.conditioning_cache_size = int(conditioning_cache_size)
        # Incremental (prefix) acoustic decoding stays OFF until its own quality
        # gate passes (plan sections 7.3/12). Enabling it is a model-quality
        # decision, not a performance flag.
        self.enable_incremental_acoustics = bool(enable_incremental_acoustics)

    @property
    def prefill_overhead(self) -> int:
        """Prefill positions that are not text tokens: conditioning + BOS BOS."""
        return self.cond_prefix_len + self.num_prefill_bos

    def max_text_tokens_for_prompt(self) -> int:
        """Largest text-token count the learned position tables actually allow."""
        return min(self.max_text_tokens, self.text_pos_table_size)


AutoConfig.register(ChatterboxMTLV3Config.model_type, ChatterboxMTLV3Config)
