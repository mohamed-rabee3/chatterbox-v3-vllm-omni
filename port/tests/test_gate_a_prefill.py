"""Gate A (part 2): prefill embedding layout == official reference, position for position.

Asserts the contracts the plan marks blocking:
  * conditioning prefix is 34 positions and identical across the CFG pair;
  * the unconditional row zeroes text CONTENT and keeps text POSITIONS;
  * the prefill ends in TWO identical BOS rows, both at learned speech pos 0;
  * the first generated token is fed at learned speech position 1.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file as load_safetensors

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K
from vllm_omni.model_executor.models.chatterbox_mtl_v3.t3 import (
    ChatterboxT3,
    WeightLoadError,
    build_prefill_embeddings,
    decode_step_embedding,
)

MODEL_DIR = (
    "/workspace/.hf_home/hub/models--ResembleAI--chatterbox/snapshots/"
    "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
)
GOLDEN = "/workspace/port/artifacts/golden"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def t3():
    """T3 custom modules (no paged backbone) loaded from the pinned V3 checkpoint."""
    model = ChatterboxT3(vllm_config=None)
    state = load_safetensors(f"{MODEL_DIR}/{K.CHECKPOINT_PROFILES['official_loader_v3']['t3']}")
    model.load_weights(state.items())
    return model.to(DEVICE).eval()


@pytest.fixture(scope="module")
def conditioning():
    out = {}
    for voice in ("ex01", "ex02"):
        ref = np.load(f"{GOLDEN}/{voice}/conditioning.npz")
        out[voice] = {k: torch.from_numpy(ref[k]).to(DEVICE) for k in ref.files}
    return out


def _cases(voice: str):
    meta = json.load(open(f"{GOLDEN}/{voice}/manifest.json"))
    return [c["case_id"] for c in meta["cases"]]


@torch.inference_mode()
def _cond_prefix(t3, cond, exaggeration=0.5):
    return t3.prepare_conditioning(
        cond["speaker_emb"],
        cond["cond_prompt_speech_tokens"],
        torch.full((1, 1, 1), exaggeration, device=DEVICE),
    )


@pytest.mark.parametrize("voice", ["ex01", "ex02"])
@torch.inference_mode()
def test_conditioning_prefix_matches_reference(t3, conditioning, voice):
    got = _cond_prefix(t3, conditioning[voice])
    ref = np.load(f"{GOLDEN}/{voice}/{_cases(voice)[0]}.npz")["cond_emb"]
    assert got.shape[1] == K.COND_PREFIX_LEN == 34
    err = float(np.abs(got.float().cpu().numpy() - ref).max())
    scale = float(np.abs(ref).max())
    assert err / scale < 1e-5, f"conditioning prefix max|diff|={err:.3e} (rel {err/scale:.3e})"


@pytest.mark.parametrize("voice", ["ex01", "ex02"])
@torch.inference_mode()
def test_prefill_matches_reference_for_every_case(t3, conditioning, voice):
    cond_prefix = _cond_prefix(t3, conditioning[voice])
    worst = 0.0
    for case_id in _cases(voice):
        ref = np.load(f"{GOLDEN}/{voice}/{case_id}.npz")
        text_ids = torch.from_numpy(ref["text_ids"]).to(DEVICE)
        for role, key in (("cond", "prefill_cond"), ("uncond", "prefill_uncond")):
            got = build_prefill_embeddings(t3, cond_prefix=cond_prefix, text_ids=text_ids, role=role)
            want = ref[key]
            assert tuple(got.shape) == want.shape, f"{case_id}/{role}: {got.shape} != {want.shape}"
            err = float(np.abs(got.float().cpu().numpy() - want).max())
            worst = max(worst, err)
            assert err < 2e-5, f"{case_id}/{role}: prefill max|diff|={err:.3e}"
    print(f"\n[{voice}] worst prefill max|diff| over {len(_cases(voice))} cases: {worst:.3e}")


@torch.inference_mode()
def test_prefill_length_is_cond_plus_text_plus_two_bos(t3, conditioning):
    cond_prefix = _cond_prefix(t3, conditioning["ex01"])
    for case_id in _cases("ex01"):
        ref = np.load(f"{GOLDEN}/ex01/{case_id}.npz")
        n_text = int(ref["text_ids"].shape[0])
        got = build_prefill_embeddings(
            t3, cond_prefix=cond_prefix, text_ids=torch.from_numpy(ref["text_ids"]).to(DEVICE)
        )
        assert got.shape[0] == K.COND_PREFIX_LEN + n_text + K.NUM_PREFILL_BOS


@torch.inference_mode()
def test_duplicate_bos_and_uncond_text_policy(t3, conditioning):
    cond_prefix = _cond_prefix(t3, conditioning["ex01"])
    ref = np.load(f"{GOLDEN}/ex01/ar_plain.npz")
    text_ids = torch.from_numpy(ref["text_ids"]).to(DEVICE)
    cond = build_prefill_embeddings(t3, cond_prefix=cond_prefix, text_ids=text_ids, role="cond")
    uncond = build_prefill_embeddings(t3, cond_prefix=cond_prefix, text_ids=text_ids, role="uncond")

    # Two identical BOS rows at the end.
    assert torch.equal(cond[-1], cond[-2])
    assert torch.equal(uncond[-1], uncond[-2])
    # BOS is speech_emb[6561] + speech_pos[0].
    expect_bos = t3.speech_emb.weight[K.START_SPEECH_TOKEN] + t3.speech_pos_emb.emb.weight[0]
    assert torch.allclose(cond[-1], expect_bos, atol=0, rtol=0)

    # The pair shares the conditioning block and the BOS rows exactly.
    assert torch.equal(cond[: K.COND_PREFIX_LEN], uncond[: K.COND_PREFIX_LEN])
    assert torch.equal(cond[-2:], uncond[-2:])

    # The uncond text block is exactly the learned text positions: content zeroed,
    # positions preserved. Not an empty-text request.
    n_text = int(text_ids.shape[0])
    lo, hi = K.COND_PREFIX_LEN, K.COND_PREFIX_LEN + n_text
    want_pos = t3.text_pos_emb.emb.weight[:n_text]
    assert torch.allclose(uncond[lo:hi], want_pos, atol=0, rtol=0)
    assert not torch.allclose(cond[lo:hi], want_pos)
    # Same length, same position indices -> the pair cannot desynchronise.
    assert cond.shape == uncond.shape


@torch.inference_mode()
def test_first_generated_token_uses_learned_speech_position_one(t3):
    ids = torch.tensor([1234], device=DEVICE)
    got = decode_step_embedding(t3, ids, torch.tensor([0], device=DEVICE))
    want = t3.speech_emb.weight[1234] + t3.speech_pos_emb.emb.weight[1]
    assert torch.allclose(got[0], want, atol=0, rtol=0)
    # ... and the k-th generated token uses position k+1.
    for k in (0, 1, 7, 99):
        got = decode_step_embedding(t3, ids, torch.tensor([k], device=DEVICE))
        want = t3.speech_emb.weight[1234] + t3.speech_pos_emb.emb.weight[k + 1]
        assert torch.allclose(got[0], want, atol=0, rtol=0)


def test_english_checkpoint_is_rejected_not_resized():
    """A 704-entry English text embedding must fail, never be resized."""
    model = ChatterboxT3(vllm_config=None)
    bad = {
        "text_emb.weight": torch.zeros(704, K.HIDDEN_SIZE),
        "speech_emb.weight": torch.zeros(K.SPEECH_VOCAB_SIZE, K.HIDDEN_SIZE),
    }
    with pytest.raises(WeightLoadError, match="different checkpoint"):
        model.load_weights(bad.items())


def test_missing_required_tensor_fails_loudly():
    model = ChatterboxT3(vllm_config=None)
    state = load_safetensors(f"{MODEL_DIR}/{K.CHECKPOINT_PROFILES['official_loader_v3']['t3']}")
    state.pop("speech_head.weight")
    with pytest.raises(WeightLoadError, match="never loaded"):
        model.load_weights(state.items())


def test_unexpected_tensor_fails_loudly():
    model = ChatterboxT3(vllm_config=None)
    state = load_safetensors(f"{MODEL_DIR}/{K.CHECKPOINT_PROFILES['official_loader_v3']['t3']}")
    state["some.unknown.tensor"] = torch.zeros(4)
    with pytest.raises(WeightLoadError, match="unexpected"):
        model.load_weights(state.items())
