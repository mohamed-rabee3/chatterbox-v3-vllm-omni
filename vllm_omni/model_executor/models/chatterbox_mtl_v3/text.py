# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Text normalization, language policy and tokenization for Chatterbox MTL V3.

The reference text path is preserved exactly -- ``punc_norm`` then the
multilingual grapheme tokenizer (lowercase, NFKD, one language marker, spaces
to ``[SPACE]``), then ``[START]`` / ``[STOP]``. It is reproduced here rather
than imported so the serving path does not depend on the upstream package, and
so every divergence would be visible in one file.

Language policy (plan section 9): ``ar`` and ``en`` are the qualified targets.
Other languages the model supports are accepted only when the deployment opts
in, and a language whose tokenizer path needs an optional third-party
normalizer is refused when that package is missing -- upstream merely logs a
warning and continues, which silently changes pronunciation.
"""

from __future__ import annotations

import importlib.util

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K


class TextPolicyError(ValueError):
    """A request rejected before any GPU work is admitted."""


# Reference `punc_norm` substitutions, in order. Reproduced verbatim: the
# double space some of them leave behind is reference behaviour and is
# preserved, because it changes the token sequence.
_PUNC_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("...", ", "),
    ("…", ", "),
    (":", ","),
    (" - ", ", "),
    (";", ", "),
    ("—", "-"),
    ("–", "-"),
    (" ,", ","),
    ("“", '"'),
    ("”", '"'),
    ("‘", "'"),
    ("’", "'"),
)

_SENTENCE_ENDERS = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}

EMPTY_TEXT_REPLACEMENT = "You need to add some text for me to talk."


def punc_norm(text: str) -> str:
    """The reference punctuation normalizer, unchanged.

    Note what it does to structured text: it turns ``:`` into ``,``, so times,
    URLs, prices and identifiers do not survive it intact. That is reference
    behaviour and is preserved for fidelity; a product spoken-form layer belongs
    *before* this function, not inside it.
    """
    if len(text) == 0:
        return EMPTY_TEXT_REPLACEMENT
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    for old, new in _PUNC_REPLACEMENTS:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not any(text.endswith(p) for p in _SENTENCE_ENDERS):
        text += "."
    return text


_LANGUAGE_ALIASES = {name.lower(): code for code, name in K.SUPPORTED_LANGUAGE_CODES.items()}
_LANGUAGE_ALIASES.update({code: code for code in K.SUPPORTED_LANGUAGE_CODES})
_LANGUAGE_ALIASES.update({"arabic": "ar", "english": "en", "ar-sa": "ar", "en-us": "en", "en-gb": "en"})


def normalize_language(language: str, config=None) -> str:
    """Canonicalize a language name/code and enforce the deployment's policy.

    The base TTS adapter's default language set does not include Arabic, so the
    Chatterbox adapter must canonicalize ``ar`` / ``Arabic`` itself rather than
    inherit a list that would reject its primary target.
    """
    if not language or not str(language).strip():
        raise TextPolicyError("language is required; Chatterbox takes one language id per call")
    key = str(language).strip().lower().replace("_", "-")
    code = _LANGUAGE_ALIASES.get(key)
    if code is None:
        code = _LANGUAGE_ALIASES.get(key.split("-")[0])
    if code is None:
        raise TextPolicyError(
            f"unsupported language {language!r}; supported: "
            f"{', '.join(sorted(K.SUPPORTED_LANGUAGE_CODES))}"
        )

    qualified = frozenset(getattr(config, "qualified_languages", None) or K.QUALIFIED_LANGUAGE_CODES)
    if code not in qualified and not bool(getattr(config, "allow_unqualified_languages", False)):
        raise TextPolicyError(
            f"language {code!r} is supported by the model but not qualified in this deployment "
            f"(qualified: {', '.join(sorted(qualified))}). Enable allow_unqualified_languages "
            f"to serve it anyway."
        )

    required = K.LANGUAGES_REQUIRING_OPTIONAL_NORMALIZER.get(code)
    if required is not None and importlib.util.find_spec(required) is None:
        # Upstream logs a warning and continues, which silently changes
        # pronunciation for that language. Refuse instead.
        raise TextPolicyError(
            f"language {code!r} needs the optional normalizer {required!r}, which is not "
            f"installed; serving it without the normalizer would change pronunciation"
        )
    return code


def build_text_token_ids(tokenizer, text: str, language: str, config=None) -> list[int]:
    """Reference text path: ``punc_norm`` -> tokenize with marker -> SOT/EOT.

    Returns the exact id sequence the reference builds, including the
    ``[START]`` prefix and ``[STOP]`` suffix.
    """
    if text is None or not str(text).strip():
        raise TextPolicyError("input text cannot be empty")
    normalized = punc_norm(str(text))
    ids = list(tokenizer.encode(normalized, language_id=language))
    ids = [K.START_TEXT_TOKEN] + ids + [K.STOP_TEXT_TOKEN]

    limit = int(getattr(config, "text_pos_table_size", K.TEXT_POS_TABLE_SIZE))
    if len(ids) > limit:
        raise TextPolicyError(
            f"text is {len(ids)} tokens; the learned text position table holds {limit}. "
            f"Split the input into clauses."
        )
    bad = [i for i in ids if not 0 <= i < K.TEXT_VOCAB_SIZE]
    if bad:
        raise TextPolicyError(f"tokenizer produced ids outside the text vocabulary: {bad[:5]}")
    return ids


def prompt_length(text_len: int) -> int:
    """Total stage-0 prompt positions for a text of ``text_len`` tokens."""
    return K.COND_PREFIX_LEN + int(text_len) + K.NUM_PREFILL_BOS
