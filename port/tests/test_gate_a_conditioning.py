"""Gate A (part 1): port conditioning == official reference conditioning.

Compares the port's `ConditioningEncoder` against the golden capture taken from
the unmodified official wrapper for the same reference wav.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import torch

from vllm_omni.model_executor.models.chatterbox_mtl_v3.conditioning import (
    ConditioningCache,
    ConditioningEncoder,
    ReferenceConditioning,
    audio_content_hash,
    conditioning_cache_key,
)

MODEL_DIR = (
    "/workspace/.hf_home/hub/models--ResembleAI--chatterbox/snapshots/"
    "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
)
GOLDEN = "/workspace/port/artifacts/golden"
VOICES = {"ex01": "/workspace/refvoices/en_ex01.wav", "ex02": "/workspace/refvoices/en_ex02.wav"}


@pytest.fixture(scope="module")
def encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ConditioningEncoder(MODEL_DIR, device=device)


@pytest.mark.parametrize("voice_id", sorted(VOICES))
def test_conditioning_matches_reference(encoder, voice_id):
    wav, sr = sf.read(VOICES[voice_id], dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    got = encoder.encode(wav, sr)
    ref = np.load(f"{GOLDEN}/{voice_id}/conditioning.npz")

    def arr(t):
        return t.detach().float().cpu().numpy()

    # Codec ids must match exactly: they are discrete and any drift is a
    # different speech prompt, not a rounding difference.
    np.testing.assert_array_equal(
        arr(got.cond_prompt_speech_tokens).astype(np.int64),
        ref["cond_prompt_speech_tokens"],
        err_msg="T3 speech prompt codec ids differ from the reference",
    )
    np.testing.assert_array_equal(
        arr(got.prompt_token).astype(np.int64),
        ref["prompt_token"],
        err_msg="S3Gen reference codec ids differ from the reference",
    )
    np.testing.assert_array_equal(
        arr(got.prompt_token_len).astype(np.int64).reshape(-1),
        ref["prompt_token_len"].reshape(-1),
    )

    # Continuous features: same computation, so differences here are float
    # reassociation only. Report the actual error rather than asserting a
    # single invented threshold.
    for name, g, r in (
        ("speaker_emb", arr(got.speaker_emb), ref["speaker_emb"]),
        ("prompt_feat", arr(got.prompt_feat), ref["prompt_feat"]),
        ("embedding", arr(got.embedding), ref["embedding"]),
    ):
        assert g.shape == r.shape, f"{name}: shape {g.shape} != {r.shape}"
        max_abs = float(np.abs(g - r).max())
        scale = float(np.abs(r).max()) or 1.0
        assert max_abs / scale < 1e-4, f"{name}: max|diff|={max_abs:.3e} (rel {max_abs/scale:.3e})"


def test_cache_key_is_content_addressed():
    a = np.zeros(2400, dtype=np.float32)
    b = a.copy()
    b[0] = 1.0
    ka = conditioning_cache_key(audio_content_hash=audio_content_hash(a, 24000),
                                checkpoint_profile="official_loader_v3")
    kb = conditioning_cache_key(audio_content_hash=audio_content_hash(b, 24000),
                                checkpoint_profile="official_loader_v3")
    assert ka != kb, "different audio content must not share a conditioning cache key"

    # Same content, different profile / preprocessing revision -> different key.
    assert ka != conditioning_cache_key(
        audio_content_hash=audio_content_hash(a, 24000), checkpoint_profile="candidate_s3gen_v3"
    )
    assert ka != conditioning_cache_key(
        audio_content_hash=audio_content_hash(a, 24000),
        checkpoint_profile="official_loader_v3",
        preprocessor_version="99",
    )


def _dummy(key: str) -> ReferenceConditioning:
    t = torch.zeros(1, 1)
    return ReferenceConditioning(
        cache_key=key, speaker_emb=t, cond_prompt_speech_tokens=t.long(),
        prompt_token=t.long(), prompt_token_len=t.long(), prompt_feat=t, embedding=t,
    )


def test_leased_entries_are_never_evicted():
    cache = ConditioningCache(capacity=2)
    held = cache.acquire_or_put(_dummy("held"))
    for i in range(10):
        cache.put(_dummy(f"other{i}"))
    assert cache.get("held") is held, "a leased voice was evicted mid-request"
    cache.release("held")
    for i in range(10, 20):
        cache.put(_dummy(f"other{i}"))
    assert cache.get("held") is None, "an unleased voice should become evictable"


def test_cache_respects_capacity_when_nothing_is_leased():
    cache = ConditioningCache(capacity=3)
    for i in range(20):
        cache.put(_dummy(f"v{i}"))
    assert cache.stats()["size"] == 3
