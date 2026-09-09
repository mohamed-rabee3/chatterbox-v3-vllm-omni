# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned constants for Chatterbox Multilingual V3.

Every value here was read out of the pinned official source or the pinned
checkpoint header; none of them are defaults to be "tuned". Changing one is a
model change and invalidates the fidelity gates.

Pinned revisions
----------------
* official ``resemble-ai/chatterbox``  ``5de7a54aa4e5e2baadb0182dde554908b48b85c2``
* HF ``ResembleAI/chatterbox``         ``5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18``
"""

from __future__ import annotations

CHATTERBOX_REPO_ID = "ResembleAI/chatterbox"
CHATTERBOX_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
UPSTREAM_SOURCE_REVISION = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"

# --- Checkpoint profiles (plan section 2.2) -------------------------------
#
# The pinned official loader selects the V3 T3 but still loads `s3gen.pt`.
# `candidate_s3gen_v3` pairs the same T3 with the V3 acoustic checkpoint and is
# a SEPARATE, independently qualified profile. There is deliberately no
# fallback between the two: the acoustic checkpoint is recorded in every
# manifest and benchmark.
CHECKPOINT_PROFILES: dict[str, dict[str, str]] = {
    "official_loader_v3": {
        "t3": "t3_mtl23ls_v3.safetensors",
        "s3gen": "s3gen.pt",
    },
    "candidate_s3gen_v3": {
        "t3": "t3_mtl23ls_v3.safetensors",
        "s3gen": "s3gen_v3.pt",
    },
}
DEFAULT_CHECKPOINT_PROFILE = "official_loader_v3"

# SHA-256 of every pinned artifact. Verified locally before first use.
ARTIFACT_SHA256: dict[str, str] = {
    "t3_mtl23ls_v3.safetensors": "5abca8321ede76f8e61f1cc0d19aea6c946b28871017ce8726f8a69203f05953",
    "s3gen.pt": "9b9ff07e60b20c136e2b1b3d7563a24604e8d2c4c267888d1ee929dd0151d2a3",
    "s3gen_v3.pt": "f7abce4b196dae2d08d9296cbebc6521b046079577643b42a19a03499d08721e",
    "s3gen_v3.safetensors": "4a46190f3dccc2230fbb3488a930bccc925862ee68f2662433dfcfe93ce6c2cb",
    "ve.pt": "4b16d836bc598509860f6fa068165a8bb5e9ac84f05582dfcf278a5a372879f1",
    "grapheme_mtl_merged_expanded_v1.json": (
        "69632f47220a788a52ce2661d096453c5655e9bf25289d89a8d832c46ee07dbf"
    ),
}

TOKENIZER_FILE = "grapheme_mtl_merged_expanded_v1.json"
VOICE_ENCODER_FILE = "ve.pt"

# --- T3 architecture ------------------------------------------------------
HIDDEN_SIZE = 1024
NUM_LAYERS = 30
NUM_HEADS = 16
NUM_KV_HEADS = 16
HEAD_DIM = 64
INTERMEDIATE_SIZE = 4096
RMS_NORM_EPS = 1e-5
ROPE_THETA = 500000.0
ROPE_SCALING = {
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}
# The backbone config's nominal 131072 is NOT a usable TTS prompt budget: the
# learned text/speech position tables below are the real constraint.
BACKBONE_MAX_POSITION_EMBEDDINGS = 131072
# Unused placeholder embedding table in the upstream backbone config. Chatterbox
# supplies its own input embeddings and speech head, so this table is never read
# during inference; the engine must never route codec ids through it.
BACKBONE_PLACEHOLDER_VOCAB_SIZE = 8

TEXT_VOCAB_SIZE = 2454
START_TEXT_TOKEN = 255
STOP_TEXT_TOKEN = 0
MAX_TEXT_TOKENS = 2048
TEXT_POS_TABLE_SIZE = 2050  # max_text_tokens + 2

SPEECH_VOCAB_SIZE = 8194  # speech head output width
START_SPEECH_TOKEN = 6561  # speech BOS
STOP_SPEECH_TOKEN = 6562  # speech EOS
CODEC_VOCAB_SIZE = 6561  # legal acoustic codec ids are 0..6560
MAX_SPEECH_TOKENS = 4096
SPEECH_POS_TABLE_SIZE = 4100  # max_speech_tokens + 2 + 2

SPEAKER_EMBED_SIZE = 256
SPEECH_COND_PROMPT_LEN = 150
PERCEIVER_OUTPUT_LEN = 32
# 1 speaker projection + 32 perceiver outputs + 1 exaggeration projection.
# Fixed for this architecture; the prefill layout below depends on it.
COND_PREFIX_LEN = 1 + PERCEIVER_OUTPUT_LEN + 1  # == 34

# Two BOS embeddings, both at learned speech position 0. This duplicate is
# reference behaviour (`prepare_input_embeds` adds one, `inference()` appends
# another); removing it changes the model's input.
NUM_PREFILL_BOS = 2

# --- Audio ----------------------------------------------------------------
S3_SR = 16_000  # speech tokenizer / voice encoder sample rate
S3GEN_SR = 24_000  # synthesis sample rate
S3_TOKEN_RATE = 25  # codec tokens per second
SAMPLES_PER_CODEC_TOKEN = S3GEN_SR // S3_TOKEN_RATE  # == 960
MEL_FRAMES_PER_CODEC_TOKEN = 2
ENC_COND_SECONDS = 6  # T3 prompt codec tokens come from the first 6 s @16 kHz
DEC_COND_SECONDS = 10  # S3Gen reference conditioning uses the first 10 s @24 kHz
ACOUSTIC_PRE_LOOKAHEAD_LEN = 3  # codec tokens dropped when finalize=False
S3GEN_N_MELS = 80  # mel channels the flow decoder produces
S3GEN_TOKEN_MEL_RATIO = 2  # mel frames per codec token

# --- incremental ("real") streaming ----------------------------------------
# Streaming can never revise audio it already sent, so the join between chunks
# is handled by HOLDING BACK this many samples and blending them with the next
# chunk's decode of the same region. 10 ms at 24 kHz: long enough to hide the
# rendering difference between two prefix decodes, short enough that it costs
# nothing anyone can perceive.
ACOUSTIC_STREAM_CROSSFADE_SAMPLES = 240

# Fewest codes a chunk may EMIT. The binding constraint is the HiFT vocoder,
# not the encoder lookahead: its reflection padding is 1024 samples per side, so
# a single code (960 samples) makes `torch.nn.functional.pad` raise
# "padding size should be less than the corresponding input dimension" and takes
# the engine down. Two codes (1920 samples) clear it. With the 3-code lookahead
# this puts the true floor for the first rung at 5 codes, not 4.
ACOUSTIC_MIN_EMIT_CODES = 2

# First streamed chunk in codec tokens. TTFA is set by this number alone, so it
# is deliberately small (1 s of audio at 25 tokens/s).
ACOUSTIC_STREAM_FIRST_BLOCK = 5

# The block is multiplied by this after each chunk. Cumulative re-decode costs
# O(n^2/B) at a fixed block -- x5.0 of the one-shot cost at 225 codes -- while
# growing it holds TTFA (set by the FIRST block) and brings the cost back to
# x1.9-x2.7. Safe for playback because the client's buffer grows faster than
# the chunks lengthen: generation runs many times faster than realtime.
ACOUSTIC_STREAM_BLOCK_GROWTH = 2.0
ACOUSTIC_STREAM_MAX_BLOCK = 400

# Rows that may share one acoustic forward pass. Only length-identical rows are
# ever grouped (padding is not isolated by this checkpoint), so the cap only
# bounds how many concurrent streams sitting at the SAME point of the decode
# ladder are served in one pass. 8 is enough for the completed-clause profile;
# streaming raises it, because there its first chunks are what batch together.
ACOUSTIC_MAX_BATCH_ROWS = 8
DEFAULT_CFM_TIMESTEPS = 10
ACOUSTIC_INFERENCE_CFG_RATE = 0.7

# --- Sampling defaults (multilingual wrapper, NOT Turbo's) ----------------
# Reference order: CFG blend -> repetition penalty over speech history ->
# temperature -> min-p -> top-p -> one multinomial draw.
DEFAULT_TEMPERATURE = 0.8
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_MIN_P = 0.05
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0  # no top-k truncation in this path
DEFAULT_CFG_WEIGHT = 0.5
DEFAULT_EXAGGERATION = 0.5
# Omni/Audex blend is g = l_u + s*(l_c - l_u); Chatterbox is g = l_c + w*(l_c - l_u).
# They agree exactly when s = 1 + w, so cfg_weight 0.5 -> cfg_scale 1.5.
DEFAULT_CFG_SCALE = 1.0 + DEFAULT_CFG_WEIGHT
# The public wrapper caps generation at 1000 new speech steps.
MAX_NEW_SPEECH_TOKENS = 1000

# --- Gross-truncation guard ------------------------------------------------
# Speech codes actually produced per text token, measured over 24 reference
# generations (12 texts x 2 voices, EN/AR/mixed/edge cases):
#
#     min 1.795   p10 2.128   median 2.547   max 5.857
#
# A generation far below that band stopped early. The threshold is set at a
# THIRD of the observed minimum so it cannot plausibly reject valid speech --
# it exists to catch gross truncation (e.g. 27 codes for a 102-token input,
# a ratio of 0.27), not to police content. It is a heuristic: it cannot prove
# an utterance is complete, only that a very short one is suspicious.
MIN_CODES_PER_TEXT_TOKEN = 0.6
# Short inputs have a high and noisy ratio dominated by fixed overhead, so the
# guard only applies above this text length.
TRUNCATION_GUARD_MIN_TEXT_TOKENS = 20

# --- Language policy ------------------------------------------------------
# Language ids accepted by the pinned multilingual tokenizer. Arabic and
# English are the deployment targets; the rest are the model's own supported
# set and are exposed rather than silently rejected.
SUPPORTED_LANGUAGE_CODES: dict[str, str] = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}
# Language ids the deployment has actually qualified end to end. Others are
# accepted only when explicitly enabled in the deploy config.
QUALIFIED_LANGUAGE_CODES = frozenset({"ar", "en"})
# Language-marker token ids in the pinned vocabulary; asserted by a gate test.
LANGUAGE_MARKER_TOKEN_IDS = {"en": 708, "ar": 721}

# Languages whose tokenizer path needs an optional third-party normalizer
# (pykakasi / dicta-onnx / russian-text-stresser / spacy-pkuseg). The upstream
# tokenizer silently skips the step when the package is absent, which would
# change pronunciation without any error, so these are refused unless the
# normalizer is importable.
LANGUAGES_REQUIRING_OPTIONAL_NORMALIZER = {
    "zh": "spacy_pkuseg",
    "ja": "pykakasi",
    "he": "dicta_onnx",
    "ko": None,  # pure-python jamo decomposition, no extra package
    "ru": "russian_text_stresser",
}

# --- Stage keys -----------------------------------------------------------
MODEL_TYPE = "chatterbox_mtl_v3"
T3_STAGE = "chatterbox_mtl_v3_t3"
S3GEN_STAGE = "chatterbox_mtl_v3_s3gen"
MODEL_ARCH = "ChatterboxMTLV3T3"

# Preprocessing revision. Any change to text normalization, reference audio
# handling or conditioning construction MUST bump this: it is folded into the
# conditioning cache key and the prefix-cache salt.
PREPROCESSOR_VERSION = "1"
