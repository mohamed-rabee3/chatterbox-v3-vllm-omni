# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Gates for the two acoustic paths the real-time profile turns on.

`test_gate_d_acoustic` proves per-row isolation for the SHIPPED defaults, but it
builds `ChatterboxS3Gen()` with `batch_vocoder=False` and the CPU watermarker.
The `chatterbox_mtl_v3_realtime.yaml` profile runs with both changed, so the
properties that matter -- a caller's audio does not depend on who else is in its
batch, and the watermark is really applied -- are re-proved here on that path.

Both changes are scheduling/placement, not model changes:

* the batched vocoder feeds `_draw` one generator per row, which draws each
  row's noise at that row's own shape, so the RNG stream per row is unchanged;
* the watermarker is the same network with the same weights, and the resampler
  that defines its signal band still runs on the CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.chatterbox_mtl_v3.conditioning import ReferenceConditioning
from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import (
    AcousticRequest,
    ChatterboxS3Gen,
)

MODEL_DIR = "/workspace/models/chatterbox-mtl-v3"
GOLDEN = "/workspace/port/artifacts/golden"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_conditioning(voice: str) -> ReferenceConditioning:
    ref = np.load(f"{GOLDEN}/{voice}/conditioning.npz")
    t = {k: torch.from_numpy(ref[k]).to(DEVICE) for k in ref.files}
    return ReferenceConditioning(
        cache_key=f"golden-{voice}",
        speaker_emb=t["speaker_emb"],
        cond_prompt_speech_tokens=t["cond_prompt_speech_tokens"].long(),
        prompt_token=t["prompt_token"].long(),
        prompt_token_len=t["prompt_token_len"].long(),
        prompt_feat=t["prompt_feat"],
        embedding=t["embedding"],
    )


@pytest.fixture(scope="module")
def batched() -> ChatterboxS3Gen:
    """Build the acoustic model, then hand back the global cuDNN TF32 setting.

    Constructing `ChatterboxS3Gen` calls `_disable_conv_tf32()`, which switches
    cuDNN TF32 off **process-wide**. That is deliberate and correct for the
    acoustic stage -- it is what stops a row's audio depending on who else is in
    its batch -- but it is not scoped to the acoustic model, and the conditioning
    encoder's convolutions are sensitive to it: measured against the golden
    capture, `ConditioningEncoder` reproduces the reference speaker embedding
    exactly (0.0) with TF32 on and drifts 2.9e-2 relative with it off, which is
    293x the Gate A threshold.

    In the shipped two-stage deploy the two never share a process -- the encoder
    runs in the input processor, the acoustic model in stage 1 -- so this is a
    hazard for single-process use, not a live defect. Inside pytest they DO
    share a process, so leaving the flag flipped would silently fail Gate A for
    whichever test file happens to sort after this one.
    """
    import vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen as s3gen_mod

    previous = torch.backends.cudnn.allow_tf32
    m = ChatterboxS3Gen(apply_watermark=False, batch_vocoder=True)
    m.load_weights_from_dir(MODEL_DIR)
    yield m.to(DEVICE).eval()
    torch.backends.cudnn.allow_tf32 = previous
    # Let the next acoustic model disable it again for its own decodes.
    s3gen_mod._TF32_DISABLED = False


def _three_rows_two_voices() -> list[AcousticRequest]:
    a, b = load_conditioning("ex01"), load_conditioning("ex02")
    codes = torch.from_numpy(
        np.load(f"{GOLDEN}/ex01/en_long.npz")["gen_ids_valid"][:97]
    ).long().to(DEVICE)
    return [
        AcousticRequest("A", codes, a, seed=11),
        AcousticRequest("B", codes, b, seed=22),
        AcousticRequest("C", codes, a, seed=33),
    ]


@torch.inference_mode()
def test_batched_vocoder_row_matches_its_solo_decode(batched):
    """Each row of a batched-vocoder pass equals that row decoded alone.

    This is the property the per-row `torch.Generator` exists to guarantee: the
    vocoder's phase and noise draws must come from the row's own stream, at the
    row's own shape, regardless of how many rows share the pass.
    """
    jobs = _three_rows_two_voices()
    assert batched.group_batches(jobs) == [[0, 1, 2]]
    solo = {j.request_id: batched.decode([j])[0] for j in jobs}
    together = {r.request_id: r for r in batched.decode_batch(jobs)}
    for rid, ref in solo.items():
        got = together[rid]
        assert got.audio.shape == ref.audio.shape, rid
        peak = float(ref.audio.abs().max())
        err = float((got.audio - ref.audio).abs().max())
        print(f"\n[batched-vocoder {rid}] batched-vs-solo max|d|={err:.3e} "
              f"({err / peak * 100:.4f}% of peak)")
        assert err < 1e-3 * peak, f"{rid}: batched vocoder drifted by {err:.3e}"
    assert not torch.equal(together["A"].audio, together["B"].audio), (
        "two different reference voices produced identical audio"
    )


@torch.inference_mode()
def test_batched_vocoder_is_order_independent(batched):
    jobs = _three_rows_two_voices()
    forward = {r.request_id: r.audio for r in batched.decode_batch(jobs)}
    reverse = {r.request_id: r.audio for r in batched.decode_batch(list(reversed(jobs)))}
    for rid, audio in forward.items():
        peak = float(audio.abs().max())
        assert float((audio - reverse[rid]).abs().max()) < 1e-3 * peak, rid


def test_watermark_many_is_bit_identical_to_the_serial_path():
    """The CPU fallback batches rows across threads; it must not change them.

    Same call, same input, one row at a time -- only the scheduling differs, so
    anything but bit-equality means a row picked up state from another row.
    """
    m = ChatterboxS3Gen.__new__(ChatterboxS3Gen)
    m.apply_watermark = True
    m._watermarker = None
    m.watermark_device = "cpu"
    m.watermark_workers = 8
    m._wm_pool = None

    rng = np.random.default_rng(7)
    for rows, secs in ((1, 0.2), (4, 0.2), (8, 1.2), (16, 0.5)):
        sigs = [(rng.standard_normal(int(24000 * secs)) * 0.1).astype(np.float32)
                for _ in range(rows)]
        serial = [m.watermark(s) for s in sigs]
        pooled = m.watermark_many(sigs)
        assert len(pooled) == rows
        for i, (want, got) in enumerate(zip(serial, pooled)):
            assert np.array_equal(want, got), f"row {i} of {rows} diverged"


def test_empty_chunk_is_passed_through_unmarked():
    """A chunk the crossfade holdback emptied must not take the engine down.

    Perth's STFT reshapes to (-1, n) and raises on a zero-length signal. There
    is no audio in an empty chunk to mark, so it passes through.
    """
    m = ChatterboxS3Gen.__new__(ChatterboxS3Gen)
    m.apply_watermark = True
    m._watermarker = None
    m.watermark_device = "cpu"
    m.watermark_workers = 4
    m._wm_pool = None

    empty = np.zeros(0, dtype=np.float32)
    assert m.watermark_many([empty])[0].size == 0
    real = (np.random.default_rng(1).standard_normal(4800) * 0.1).astype(np.float32)
    out = m.watermark_many([empty, real, empty])
    assert out[0].size == 0 and out[2].size == 0
    assert out[1].size == real.size and not np.array_equal(out[1], real)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_watermark_tracks_the_cpu_watermark():
    """Running Perth on the acoustic device is a placement change only.

    The bar is the watermark's OWN perturbation of the waveform: the CPU/GPU
    difference must be a tiny fraction of it, otherwise the placement is
    changing what gets embedded rather than where it is computed.
    """
    import perth

    cpu = perth.PerthImplicitWatermarker(device="cpu")
    gpu = perth.PerthImplicitWatermarker(device="cuda")
    rng = np.random.default_rng(5)
    for secs in (0.2, 1.6, 6.4):
        sig = (rng.standard_normal(int(24000 * secs)) * 0.08).astype(np.float32)
        with torch.inference_mode():
            on_cpu = cpu.apply_watermark(sig, sample_rate=24000)
            on_gpu = gpu.apply_watermark(sig, sample_rate=24000)
        assert on_cpu.shape == on_gpu.shape
        delta = float(np.abs(on_cpu - on_gpu).max())
        strength = float(np.abs(on_cpu - sig).max())
        print(f"\n[perth {secs}s] cpu-vs-gpu={delta:.3e} watermark={strength:.3e} "
              f"ratio={delta / strength:.2e}")
        assert delta < 1e-3 * strength, (
            f"GPU watermark differs from CPU by {delta:.3e}, "
            f"{delta / strength:.1%} of the watermark itself"
        )
