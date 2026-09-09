"""Gate A (tokenizer half): text normalization + tokenization == the reference.

Compared against the normalized text and token ids captured from the official
runner for all 12 EN/AR/mixed/edge cases.
"""

from __future__ import annotations

import json

import pytest

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K
from vllm_omni.model_executor.models.chatterbox_mtl_v3.text import (
    TextPolicyError,
    build_text_token_ids,
    normalize_language,
    prompt_length,
    punc_norm,
)
from vllm_omni.transformers_utils.configs.chatterbox_mtl_v3 import ChatterboxMTLV3Config

MODEL_DIR = (
    "/workspace/.hf_home/hub/models--ResembleAI--chatterbox/snapshots/"
    "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
)
GOLDEN = "/workspace/port/artifacts/golden/ex01"


@pytest.fixture(scope="module")
def tokenizer():
    from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.tokenizers import MTLTokenizer

    return MTLTokenizer(f"{MODEL_DIR}/{K.TOKENIZER_FILE}")


@pytest.fixture(scope="module")
def cases():
    return json.load(open(f"{GOLDEN}/manifest.json"))["cases"]


@pytest.fixture(scope="module")
def config():
    return ChatterboxMTLV3Config()


def test_normalized_text_matches_reference(cases):
    for c in cases:
        assert punc_norm(c["text"]) == c["normalized_text"], (
            f"{c['case_id']}: {punc_norm(c['text'])!r} != {c['normalized_text']!r}"
        )


def test_token_ids_match_reference_exactly(tokenizer, cases, config):
    for c in cases:
        got = build_text_token_ids(tokenizer, c["text"], c["language"], config)
        assert got == c["text_ids"], f"{c['case_id']}: token ids differ from the reference"
        assert got[0] == K.START_TEXT_TOKEN and got[-1] == K.STOP_TEXT_TOKEN
        # Exactly one language marker, immediately after [START].
        assert got[1] == K.LANGUAGE_MARKER_TOKEN_IDS[c["language"]]


def test_language_marker_ids_are_the_pinned_ones(tokenizer):
    vocab = tokenizer.tokenizer.get_vocab()
    assert vocab["[en]"] == K.LANGUAGE_MARKER_TOKEN_IDS["en"] == 708
    assert vocab["[ar]"] == K.LANGUAGE_MARKER_TOKEN_IDS["ar"] == 721
    assert vocab["[START]"] == K.START_TEXT_TOKEN
    assert vocab["[STOP]"] == K.STOP_TEXT_TOKEN
    assert tokenizer.tokenizer.get_vocab_size() == K.TEXT_VOCAB_SIZE == 2454


def test_prompt_length_matches_reference_prefill(cases):
    for c in cases:
        assert prompt_length(len(c["text_ids"])) == c["prefill_shape"][0]


def test_arabic_is_a_first_class_language(config):
    assert normalize_language("ar", config) == "ar"
    assert normalize_language("Arabic", config) == "ar"
    assert normalize_language("AR", config) == "ar"
    assert normalize_language("ar-SA", config) == "ar"
    assert normalize_language("English", config) == "en"


def test_unqualified_language_is_refused_unless_enabled(config):
    with pytest.raises(TextPolicyError, match="not qualified"):
        normalize_language("es", config)
    permissive = ChatterboxMTLV3Config(allow_unqualified_languages=True)
    assert normalize_language("es", permissive) == "es"


def test_language_needing_a_missing_normalizer_is_refused(config):
    """Upstream warns and continues, silently changing pronunciation."""
    permissive = ChatterboxMTLV3Config(allow_unqualified_languages=True)
    import importlib.util

    if importlib.util.find_spec("pykakasi") is None:
        with pytest.raises(TextPolicyError, match="pykakasi"):
            normalize_language("ja", permissive)
    else:
        assert normalize_language("ja", permissive) == "ja"


def test_unknown_language_and_empty_input_are_refused(config, tokenizer):
    with pytest.raises(TextPolicyError, match="unsupported language"):
        normalize_language("klingon", config)
    with pytest.raises(TextPolicyError, match="language is required"):
        normalize_language("", config)
    with pytest.raises(TextPolicyError, match="cannot be empty"):
        build_text_token_ids(tokenizer, "   ", "en", config)


def test_over_long_text_is_refused_rather_than_truncated(tokenizer, config):
    with pytest.raises(TextPolicyError, match="learned text position table"):
        build_text_token_ids(tokenizer, "hello " * 3000, "en", config)


def test_punc_norm_preserves_reference_quirks():
    # Colon -> comma, and the double space the substitution leaves behind.
    assert punc_norm("Wait... what?! - okay; fine: let's go.") == "Wait,  what?!, okay,  fine, let's go."
    # A trailing full stop is added when there is no sentence ender.
    assert punc_norm("   spaced    out   text   ") == "spaced out text."
    assert punc_norm("") == "You need to add some text for me to talk."


def test_same_text_two_languages_produces_different_ids(tokenizer, config):
    en = build_text_token_ids(tokenizer, "hello", "en", config)
    ar = build_text_token_ids(tokenizer, "hello", "ar", config)
    assert en != ar and en[1] != ar[1]
