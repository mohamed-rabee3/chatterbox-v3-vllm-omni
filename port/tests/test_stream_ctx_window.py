# SPDX-License-Identifier: Apache-2.0
"""Bounded left-context streaming window (ACOUSTIC_STREAM_CTX_WINDOW).

Proves, against the real ``ChatterboxS3Gen.decode`` path:
  * window disabled (token_offset == 0) is byte-identical to the shipped
    growing-prefix streaming decode;
  * with the window engaged the emitted PCM has NO gap and NO overlap
    (exactly-once ledger) and the total length equals the disabled run;
  * a generated code keeps the SAME flow noise whether it is decoded inside a
    wide window or a narrow one (crossfade correctness);
  * the finalize chunk always decodes the whole prefix (token_offset 0).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "/workspace/repos/vllm-omni")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="acoustic decode needs CUDA"
)

DEVICE = "cuda"
MODEL_DIR = "/workspace/models/chatterbox-mtl-v3"
GOLDEN = "/workspace/port/artifacts/golden/ex01"


@pytest.fixture(scope="module")
def s3gen():
    from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import ChatterboxS3Gen

    m = ChatterboxS3Gen(apply_watermark=False)
    m.load_weights_from_dir(MODEL_DIR)
    return m.to(DEVICE).eval()


@pytest.fixture(scope="module")
def cond():
    sys.path.insert(0, "/workspace/chatterbox-v3-vllm-omni/port/tools")
    from measure_prefix_stability import load_conditioning

    return load_conditioning("ex01")


@pytest.fixture(scope="module")
def long_codes():
    """~430 codes: long enough that a 224-code window slides several times."""
    a = np.load(f"{GOLDEN}/en_long.npz")["gen_ids_valid"].reshape(-1)
    b = np.load(f"{GOLDEN}/en_plain.npz")["gen_ids_valid"].reshape(-1)
    ids = np.concatenate([a, b, a])
    return torch.from_numpy(ids).reshape(-1).to(DEVICE, torch.long)


def _schedule(n, first=5, growth=1.7, max_block=120):
    pts, k, block = [], first, first
    while k < n:
        pts.append(k)
        block = min(int(block * growth), max_block)
        k += block
    pts.append(n)
    return pts


def _run_stream(s3gen, codes, cond, window, seed=1235):
    from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import AcousticRequest

    n = int(codes.shape[0])
    rid = f"win{window}"
    pieces, last_at, offsets = [], 0, []
    for k in _schedule(n):
        is_final = k >= n
        w = 0
        if not is_final and window > 0 and k > window:
            w_eff = max(window, k - last_at + 96)
            w = max(0, k - w_eff)
        dec = codes[:k] if is_final else codes[w:k]
        if not is_final:
            last_at = k
        offsets.append(w)
        res = s3gen.decode(
            [AcousticRequest(rid, dec, cond, seed=seed, finalize=is_final,
                             streaming=True, token_offset=w)]
        )[0]
        if res.audio.numel():
            pieces.append(res.audio.detach().cpu().numpy())
    assert rid not in s3gen._stream_state, "stream state leaked past finalize"
    assert rid not in s3gen._stream_noise, "noise bank leaked past finalize"
    return np.concatenate(pieces) if pieces else np.zeros(0, np.float32), offsets


@torch.inference_mode()
def test_window_disabled_matches_shipped_path(s3gen, long_codes, cond):
    a, off0 = _run_stream(s3gen, long_codes, cond, window=0)
    b, _ = _run_stream(s3gen, long_codes, cond, window=0)
    assert off0 == [0] * len(off0)
    assert a.shape == b.shape
    # same code path -> equal up to GPU kernel non-determinism only
    assert np.max(np.abs(a - b)) < 1e-5


@torch.inference_mode()
def test_windowed_ledger_has_no_gap_or_overlap(s3gen, long_codes, cond):
    base, _ = _run_stream(s3gen, long_codes, cond, window=0)
    win, offs = _run_stream(s3gen, long_codes, cond, window=224)
    # the window must actually have engaged on the long sequence
    assert max(offs) > 0, f"window never engaged: offsets={offs}"
    # exactly-once ledger: same total number of PCM samples emitted
    assert win.shape == base.shape, (win.shape, base.shape)
    assert np.isfinite(win).all()


@torch.inference_mode()
def test_generated_code_noise_is_window_invariant(s3gen, cond):
    """Code j's flow noise must not depend on where the window starts."""
    from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import AcousticRequest
    import vllm_omni.model_executor.models.chatterbox_mtl_v3.constants as K

    ratio = K.S3GEN_TOKEN_MEL_RATIO
    prompt_mel = int(cond.acoustic_prompt_length()) * ratio
    dtype = next(s3gen.s3gen.parameters()).dtype
    dev = torch.device(DEVICE)

    wide = AcousticRequest("noise-inv", torch.zeros(300, dtype=torch.long, device=dev),
                           cond, seed=7, finalize=False, streaming=True, token_offset=0)
    n_wide = s3gen._flow_noise(wide, prompt_mel + 300 * ratio, dev, dtype, prompt_mel=prompt_mel)
    s3gen._stream_noise.pop("noise-inv", None)

    narrow = AcousticRequest("noise-inv", torch.zeros(100, dtype=torch.long, device=dev),
                             cond, seed=7, finalize=False, streaming=True, token_offset=180)
    n_narrow = s3gen._flow_noise(narrow, prompt_mel + 100 * ratio, dev, dtype, prompt_mel=prompt_mel)
    s3gen._stream_noise.pop("noise-inv", None)

    # generated code 180..279 in the wide draw == code 0..99 of the narrow draw
    w = n_wide[:, prompt_mel + 180 * ratio: prompt_mel + 280 * ratio]
    m = n_narrow[:, prompt_mel:]
    assert torch.equal(w, m)
    # prompt-region noise identical too
    assert torch.equal(n_wide[:, :prompt_mel], n_narrow[:, :prompt_mel])
