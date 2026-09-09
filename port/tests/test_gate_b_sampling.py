"""Gate B/C (sampling half): the guided-sampling math equals the reference loop.

The reference order is CFG blend -> repetition penalty over the speech history
-> temperature -> min-p -> top-p -> one draw. vLLM 0.28 runs a
non-argmax-invariant processor before penalties and before temperature, and
min-p as an argmax-invariant processor after temperature, so putting the blend
and the speech-only penalty in this processor reproduces that order exactly.

These tests use the *real* HuggingFace processors as the oracle.
"""

from __future__ import annotations

import pytest
import torch
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K
from vllm_omni.model_executor.models.chatterbox_mtl_v3.sampling import (
    COND,
    POLICY_HARDENED,
    POLICY_REFERENCE,
    UNCOND,
    ChatterboxCFGError,
    ChatterboxCFGLogitsProcessor,
    cfg_scale_from_weight,
    clear_cfg_failures,
    failed_cfg_pairs,
    legal_speech_token_mask,
)

V = K.SPEECH_VOCAB_SIZE


def make_proc() -> ChatterboxCFGLogitsProcessor:
    clear_cfg_failures()
    return ChatterboxCFGLogitsProcessor(None, torch.device("cpu"), False)


def params(role: str, pair_id: str, *, cfg_weight=K.DEFAULT_CFG_WEIGHT,
           penalty=K.DEFAULT_REPETITION_PENALTY, policy=POLICY_REFERENCE) -> SamplingParams:
    return SamplingParams(
        temperature=K.DEFAULT_TEMPERATURE,
        repetition_penalty=1.0,
        extra_args={
            "cfg_role": role,
            "cfg_pair_id": pair_id,
            "cfg_scale": cfg_scale_from_weight(cfg_weight),
            "chatterbox_repetition_penalty": penalty,
            "chatterbox_policy": policy,
        },
    )


def add_pair(proc, pair_id, cond_idx, uncond_idx, cond_hist, uncond_hist, **kw):
    proc.update_state(
        BatchUpdate(
            batch_size=max(cond_idx, uncond_idx) + 1,
            removed=[],
            added=[
                (cond_idx, params(COND, pair_id, **kw), [], cond_hist),
                (uncond_idx, params(UNCOND, pair_id, **kw), [], uncond_hist),
            ],
            moved=[],
        )
    )


def reference_step(cond, uncond, history, *, cfg_weight, penalty, temperature, min_p, top_p):
    """The pinned reference decode step, using HF's own processors."""
    cfg = torch.as_tensor(cfg_weight, dtype=cond.dtype)
    logits = cond + cfg * (cond - uncond)
    logits = logits.unsqueeze(0)
    ids = torch.tensor([history], dtype=torch.long)
    logits = RepetitionPenaltyLogitsProcessor(penalty=float(penalty))(ids, logits)
    if temperature != 1.0:
        logits = logits / temperature
    logits = MinPLogitsWarper(min_p=min_p)(ids, logits)
    logits = TopPLogitsWarper(top_p=top_p)(ids, logits)
    return logits.squeeze(0)


def engine_tail(logits, *, temperature, min_p, top_p):
    """What the engine does after our processor: temperature, min-p, top-p."""
    out = logits.unsqueeze(0) / temperature
    out = MinPLogitsWarper(min_p=min_p)(torch.zeros(1, 1, dtype=torch.long), out)
    out = TopPLogitsWarper(top_p=top_p)(torch.zeros(1, 1, dtype=torch.long), out)
    return out.squeeze(0)


@pytest.mark.parametrize("n_generated", [0, 1, 5, 40])
def test_matches_reference_decode_step(n_generated):
    torch.manual_seed(7 + n_generated)
    cond = torch.randn(V, dtype=torch.float32)
    uncond = torch.randn(V, dtype=torch.float32)
    generated = torch.randint(0, K.CODEC_VOCAB_SIZE, (n_generated,)).tolist()

    proc = make_proc()
    add_pair(proc, "p0", 0, 1, list(generated), list(generated))

    logits = torch.stack([cond.clone(), uncond.clone()])
    out = proc.apply(logits)
    got = engine_tail(
        out[0], temperature=K.DEFAULT_TEMPERATURE, min_p=K.DEFAULT_MIN_P, top_p=K.DEFAULT_TOP_P
    )

    # The reference history starts with speech BOS and then the generated ids.
    want = reference_step(
        cond, uncond,
        [K.START_SPEECH_TOKEN] + generated,
        cfg_weight=K.DEFAULT_CFG_WEIGHT,
        penalty=K.DEFAULT_REPETITION_PENALTY,
        temperature=K.DEFAULT_TEMPERATURE,
        min_p=K.DEFAULT_MIN_P,
        top_p=K.DEFAULT_TOP_P,
    )
    finite = torch.isfinite(want)
    assert torch.equal(finite, torch.isfinite(got)), "different -inf support than the reference"
    assert torch.allclose(got[finite], want[finite], atol=1e-5, rtol=1e-5)


def test_cfg_scale_conversion_is_the_same_function():
    torch.manual_seed(1)
    cond, uncond = torch.randn(64), torch.randn(64)
    w = 0.5
    chatterbox = cond + w * (cond - uncond)
    omni = uncond + cfg_scale_from_weight(w) * (cond - uncond)
    assert torch.allclose(chatterbox, omni, atol=1e-6)
    assert cfg_scale_from_weight(0.5) == 1.5


def test_both_pair_rows_receive_identical_logits():
    proc = make_proc()
    add_pair(proc, "p0", 0, 1, [], [])
    torch.manual_seed(3)
    logits = torch.randn(2, V)
    out = proc.apply(logits)
    assert torch.equal(out[0], out[1]), "pair rows must sample from identical logits"


def test_repetition_penalty_ignores_prompt_and_uses_bos_seeded_speech_history():
    proc = make_proc()
    # Prompt token ids are deliberately non-empty and would change the result
    # if the penalty ever looked at them.
    prompt = list(range(100, 140))
    proc.update_state(
        BatchUpdate(
            batch_size=2, removed=[],
            added=[(0, params(COND, "p"), prompt, [7]), (1, params(UNCOND, "p"), prompt, [7])],
            moved=[],
        )
    )
    logits = torch.zeros(2, V)
    logits[:, :] = 1.0
    out = proc.apply(logits.clone())
    penalised = {int(i) for i in torch.nonzero(out[0] != 1.0).flatten()}
    assert penalised == {K.START_SPEECH_TOKEN, 7}, (
        "penalty must cover exactly [speech BOS] + generated ids, not the prompt"
    )


def test_nonfinite_raw_logits_are_rejected_not_concealed():
    proc = make_proc()
    add_pair(proc, "p0", 0, 1, [], [])
    logits = torch.zeros(2, V)
    logits[0, 5] = float("nan")
    with pytest.raises(ChatterboxCFGError, match="non-finite"):
        proc.apply(logits)


def test_hardened_mask_is_applied_after_the_blend_and_never_makes_nan():
    proc = make_proc()
    add_pair(proc, "p0", 0, 1, [], [], policy=POLICY_HARDENED)
    torch.manual_seed(11)
    logits = torch.randn(2, V)
    out = proc.apply(logits)
    legal = legal_speech_token_mask()
    assert not torch.isnan(out).any(), "hardened masking produced NaN (-inf minus -inf)"
    assert torch.isinf(out[0][~legal]).all() and (out[0][~legal] < 0).all()
    assert torch.isfinite(out[0][legal]).all()
    # Speech BOS is never a legal generated output.
    assert not legal[K.START_SPEECH_TOKEN]
    assert legal[K.STOP_SPEECH_TOKEN]


def test_reference_policy_does_not_mask_the_non_codec_domain():
    """The reference loop does NOT hard-mask; `hardened` is a different distribution."""
    proc = make_proc()
    add_pair(proc, "p0", 0, 1, [], [], policy=POLICY_REFERENCE)
    torch.manual_seed(11)
    out = proc.apply(torch.randn(2, V))
    assert torch.isfinite(out[0]).all()


def test_lost_companion_fails_the_request_instead_of_going_unguided():
    proc = make_proc()
    proc.update_state(
        BatchUpdate(batch_size=1, removed=[],
                    added=[(0, params(COND, "lonely"), [], [])], moved=[])
    )
    torch.manual_seed(5)
    logits = torch.randn(1, V)
    out = proc.apply(logits.clone())
    assert "lonely" in failed_cfg_pairs(), "a broken pair must be recorded as a failure"
    assert int(out[0].argmax()) == K.STOP_SPEECH_TOKEN, "a broken pair must terminate, not continue"
    assert torch.isinf(out[0][: K.STOP_SPEECH_TOKEN]).all()


def test_pair_survives_row_moves_and_swaps():
    proc = make_proc()
    add_pair(proc, "p0", 0, 1, [3], [3])
    # Persistent-batch compaction moves row 1 -> row 5.
    proc.update_state(
        BatchUpdate(batch_size=6, removed=[], added=[], moved=[(1, 5, MoveDirectionality.UNIDIRECTIONAL)])
    )
    torch.manual_seed(9)
    logits = torch.randn(6, V)
    out = proc.apply(logits.clone())
    assert torch.equal(out[0], out[5]), "the pair must follow its rows after a move"
    assert not failed_cfg_pairs()

    # ... and after a swap.
    proc.update_state(
        BatchUpdate(batch_size=6, removed=[], added=[], moved=[(0, 5, MoveDirectionality.SWAP)])
    )
    out = proc.apply(torch.randn(6, V))
    assert torch.equal(out[0], out[5])
    assert not failed_cfg_pairs()


def test_removed_row_does_not_leak_history_into_a_new_request():
    proc = make_proc()
    add_pair(proc, "p0", 0, 1, [11, 12, 13], [11, 12, 13])
    proc.apply(torch.ones(2, V))
    proc.update_state(BatchUpdate(batch_size=2, removed=[0, 1], added=[], moved=[]))
    add_pair(proc, "p1", 0, 1, [], [])
    out = proc.apply(torch.ones(2, V))
    penalised = {int(i) for i in torch.nonzero(out[0] != 1.0).flatten()}
    assert penalised == {K.START_SPEECH_TOKEN}, "a new request inherited the old row's history"


def test_history_rewind_after_preemption_rebuilds_the_penalty_mask():
    proc = make_proc()
    hist = [21, 22, 23]
    add_pair(proc, "p0", 0, 1, hist, hist)
    proc.apply(torch.ones(2, V))
    hist.clear()  # recompute rewound the request's outputs
    out = proc.apply(torch.ones(2, V))
    penalised = {int(i) for i in torch.nonzero(out[0] != 1.0).flatten()}
    assert penalised == {K.START_SPEECH_TOKEN}


def test_validate_params_rejects_double_repetition_penalty():
    p = SamplingParams(repetition_penalty=1.2, extra_args={
        "cfg_role": COND, "cfg_pair_id": "x", "cfg_scale": 1.5})
    with pytest.raises(ValueError, match="must be exactly 1.0"):
        ChatterboxCFGLogitsProcessor.validate_params(p)


def test_validate_params_rejects_missing_pair_id_and_bad_scale():
    with pytest.raises(ValueError, match="cfg_pair_id"):
        ChatterboxCFGLogitsProcessor.validate_params(
            SamplingParams(repetition_penalty=1.0, extra_args={"cfg_role": COND, "cfg_scale": 1.5})
        )
    with pytest.raises(ValueError, match="cfg_scale"):
        ChatterboxCFGLogitsProcessor.validate_params(
            SamplingParams(repetition_penalty=1.0,
                           extra_args={"cfg_role": COND, "cfg_pair_id": "x", "cfg_scale": 0.5})
        )
