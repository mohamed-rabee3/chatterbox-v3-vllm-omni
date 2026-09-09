"""Gate E -- incremental ("real") acoustic streaming.

The contract these pin is deliberately NOT "streamed equals one-shot". The
token encoder is bidirectional, so re-decoding a prefix with more context
renders it slightly differently; streaming returns a different-but-equally-
valid rendering. What must hold is that the stream is emit-once, seam-free,
correctly terminated and exactly as long as the one-shot decode.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, "/workspace/port/tools")

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K
from vllm_omni.model_executor.models.chatterbox_mtl_v3.conditioning import ReferenceConditioning
from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import (
    AcousticRequest,
    ChatterboxS3Gen,
)
from vllm_omni.model_executor.models.chatterbox_mtl_v3.streaming import CodecCursor

MODEL_DIR = (
    "/workspace/.hf_home/hub/models--ResembleAI--chatterbox/snapshots/"
    "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
)
GOLDEN = "/workspace/port/artifacts/golden"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def s3gen():
    low_latency = os.environ.get("CHATTERBOX_TEST_LOW_LATENCY") == "1"
    m = ChatterboxS3Gen(
        apply_watermark=False,
        cfm_timesteps=4 if low_latency else K.DEFAULT_CFM_TIMESTEPS,
        flow_cudagraphs=low_latency,
        estimator_dtype=os.environ.get("CHATTERBOX_TEST_ESTIMATOR_DTYPE", "float32"),
        vocoder_cudagraphs=os.environ.get("CHATTERBOX_TEST_VOCODER_GRAPHS") == "1",
        materialize_vocoder_weights=os.environ.get("CHATTERBOX_TEST_VOCODER_GRAPHS") == "1",
        compile_estimator=low_latency,
    )
    m.load_weights_from_dir(MODEL_DIR)
    m = m.to(DEVICE).eval()
    m.warm_compiled_estimator()
    return m


def load_conditioning(voice: str = "ex01") -> ReferenceConditioning:
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


def golden_codes(case: str = "en_plain", voice: str = "ex01") -> torch.Tensor:
    cap = np.load(f"{GOLDEN}/{voice}/{case}.npz")
    return torch.from_numpy(cap["gen_ids_valid"]).reshape(-1).to(DEVICE, torch.long)


def schedule(n: int, first: int, growth: float, max_block: int) -> list[int]:
    points, k, block = [], first, first
    while k < n:
        points.append(k)
        block = min(int(block * growth), max_block) if growth > 1.0 else block
        k += block
    points.append(n)
    return points


def stream(s3gen, codes, cond, seed=1235, first=None, growth=None):
    first = K.ACOUSTIC_STREAM_FIRST_BLOCK if first is None else first
    growth = K.ACOUSTIC_STREAM_BLOCK_GROWTH if growth is None else growth
    n = int(codes.shape[0])
    rid = "stream-test"
    pieces = []
    for k in schedule(n, first, growth, K.ACOUSTIC_STREAM_MAX_BLOCK):
        res = s3gen.decode([
            AcousticRequest(rid, codes[:k], cond, seed=seed,
                            finalize=(k >= n), streaming=True)
        ])[0]
        if res.audio.numel():
            pieces.append(res.audio.cpu().numpy())
    return np.concatenate(pieces), pieces, rid


# ---------------------------------------------------------------------------
# waveform contract
# ---------------------------------------------------------------------------
@torch.inference_mode()
def test_streamed_total_length_equals_one_shot(s3gen):
    """Emit-once accounting must be exact: no sample dropped, none sent twice."""
    cond, codes = load_conditioning(), golden_codes()
    one = s3gen.decode([AcousticRequest("one", codes, cond, seed=1235)])[0].audio
    streamed, _, _ = stream(s3gen, codes, cond)
    assert len(streamed) == int(one.numel())


@torch.inference_mode()
def test_streaming_emits_first_audio_from_the_first_block(s3gen):
    """TTFA must depend on the first block, not on the utterance length."""
    cond, codes = load_conditioning(), golden_codes("en_long")
    _, pieces, _ = stream(s3gen, codes, cond)
    first_block_samples = (K.ACOUSTIC_STREAM_FIRST_BLOCK - K.ACOUSTIC_PRE_LOOKAHEAD_LEN) * 960
    # The first emission covers the first block minus the lookahead and minus
    # the held-back crossfade -- never the whole utterance.
    assert 0 < len(pieces[0]) <= first_block_samples
    assert len(pieces) > 1, "a long utterance must arrive in more than one chunk"


@torch.inference_mode()
def test_streamed_audio_has_no_seam(s3gen):
    """No chunk join may show up as a click.

    A click is a first-difference far outside the signal's own range, so the
    streamed waveform's largest jump is compared against the one-shot decode's.
    """
    cond, codes = load_conditioning(), golden_codes()
    one = s3gen.decode([AcousticRequest("one", codes, cond, seed=1235)])[0].audio.cpu().numpy()
    streamed, _, _ = stream(s3gen, codes, cond)
    jump_streamed = float(np.abs(np.diff(streamed)).max())
    jump_oneshot = float(np.abs(np.diff(one)).max())
    assert jump_streamed < 3.0 * jump_oneshot, (
        f"streamed max jump {jump_streamed:.4f} vs one-shot {jump_oneshot:.4f}"
    )


@torch.inference_mode()
def test_chunk_below_the_vocoder_floor_emits_nothing(s3gen):
    """A chunk that would emit one code must be a no-op, not an engine crash.

    The HiFT vocoder reflection-pads 1024 samples per side, so a single emitted
    code (960 samples) makes `pad` raise and takes the whole engine down. Two
    codes (1920 samples) clear it. With the 3-code encoder lookahead that puts
    the floor for the first rung at 5 codes, and a first rung of 4 -- which
    looks legal, being lookahead+1 -- killed the server under load until this
    was pinned.
    """
    cond, codes = load_conditioning(), golden_codes()
    n_emit = K.ACOUSTIC_MIN_EMIT_CODES - 1
    res = s3gen.decode([
        AcousticRequest("floor", codes[: K.ACOUSTIC_PRE_LOOKAHEAD_LEN + n_emit], cond,
                        seed=1, finalize=False, streaming=True)
    ])[0]
    assert res.audio.numel() == 0

    # One more code clears the floor and must produce audio.
    ok = s3gen.decode([
        AcousticRequest("floor-ok",
                        codes[: K.ACOUSTIC_PRE_LOOKAHEAD_LEN + K.ACOUSTIC_MIN_EMIT_CODES],
                        cond, seed=1, finalize=False, streaming=True)
    ])[0]
    assert ok.audio.numel() > 0


def test_first_rung_clears_the_vocoder_floor():
    """The shipped first rung must be decodable, not merely > the lookahead."""
    assert (K.ACOUSTIC_STREAM_FIRST_BLOCK
            >= K.ACOUSTIC_PRE_LOOKAHEAD_LEN + K.ACOUSTIC_MIN_EMIT_CODES)


@torch.inference_mode()
def test_streaming_state_is_released_on_finalize(s3gen):
    """The held-back tail must not outlive the request."""
    cond, codes = load_conditioning(), golden_codes()
    _, _, rid = stream(s3gen, codes, cond)
    assert rid not in s3gen._stream_state


@torch.inference_mode()
def test_cancelled_stream_state_is_reclaimable(s3gen):
    """A barge-in leaves a tail behind; it must be droppable."""
    cond, codes = load_conditioning(), golden_codes()
    rid = "cancelled-stream"
    s3gen.decode([
        AcousticRequest(rid, codes[:25], cond, seed=1235, finalize=False, streaming=True)
    ])
    assert rid in s3gen._stream_state
    s3gen.release_stream_state([rid])
    assert rid not in s3gen._stream_state


@torch.inference_mode()
def test_non_final_chunk_shorter_than_lookahead_emits_nothing(s3gen):
    """Too few codes to clear the encoder lookahead is a no-op, not an error.

    Chunks are cumulative, so these codes arrive again in the next chunk and
    nothing is lost by emitting silence now. Raising here would kill the engine
    over an ordinary early-stream state.
    """
    cond, codes = load_conditioning(), golden_codes()
    res = s3gen.decode([
        AcousticRequest("tiny", codes[:2], cond, seed=1,
                        finalize=False, streaming=True)
    ])[0]
    assert res.audio.numel() == 0
    assert not res.watermarked


@torch.inference_mode()
def test_one_shot_path_is_unchanged_by_the_streaming_flag(s3gen):
    """The qualified non-streaming rendering must not move.

    `streaming=True` switches the flow noise to a fixed bank, which is a
    different (equally valid) rendering. A one-shot request must keep the draw
    the non-streaming gates were measured with, so the two must NOT match.
    """
    cond, codes = load_conditioning(), golden_codes()
    a = s3gen.decode([AcousticRequest("r", codes, cond, seed=7)])[0].audio
    b = s3gen.decode([AcousticRequest("r", codes, cond, seed=7)])[0].audio
    # Repeat decodes agree to within one 16-bit LSB, not bit-exactly: GPU
    # kernel scheduling leaves ~1e-10 of float noise. That is the same bound
    # the live determinism check uses.
    one_lsb = 1.0 / 32768.0
    assert float((a - b).abs().max()) < one_lsb, "one-shot decode must be reproducible"

    c = s3gen.decode([
        AcousticRequest("r", codes, cond, seed=7, streaming=True)
    ])[0].audio
    assert float((a - c[: a.numel()]).abs().max()) > one_lsb, (
        "streaming must use the fixed noise bank, not the one-shot draw"
    )


# ---------------------------------------------------------------------------
# cursor contract
# ---------------------------------------------------------------------------
def test_take_prefix_returns_cumulative_codes_and_grows_the_block():
    cursor = CodecCursor(request_id="r")
    cursor.observe(list(range(1, 26)))
    first = cursor.take_prefix(first_block=25, growth=2.0, max_block=400, lookahead=3)
    assert first is not None
    codes, final = first
    assert codes == list(range(1, 26)) and final is False

    # Next decode point must be 25 + 50, so 25 more codes is NOT yet enough.
    cursor.observe(list(range(1, 51)))
    assert cursor.take_prefix(first_block=25, growth=2.0, max_block=400, lookahead=3) is None

    cursor.observe(list(range(1, 81)))
    second = cursor.take_prefix(first_block=25, growth=2.0, max_block=400, lookahead=3)
    assert second is not None
    codes2, final2 = second
    assert codes2 == list(range(1, 81)), "chunks must be CUMULATIVE, not deltas"
    assert final2 is False


def test_take_prefix_flush_is_final_and_happens_once():
    cursor = CodecCursor(request_id="r")
    cursor.observe(list(range(1, 31)) + [K.STOP_SPEECH_TOKEN])
    out = cursor.take_prefix(
        first_block=25, growth=2.0, max_block=400, lookahead=3, force_flush=True
    )
    assert out is not None
    codes, final = out
    assert final is True and codes == list(range(1, 31))
    assert cursor.take_prefix(
        first_block=25, growth=2.0, max_block=400, lookahead=3, force_flush=True
    ) is None


def test_take_prefix_waits_for_more_than_the_lookahead():
    cursor = CodecCursor(request_id="r")
    cursor.observe([1, 2])
    assert cursor.take_prefix(
        first_block=1, growth=1.0, max_block=400, lookahead=3
    ) is None

@torch.inference_mode()
def test_cached_stream_noise_is_reused_and_released(s3gen):
    cond, codes = load_conditioning(), golden_codes()
    req = AcousticRequest('noise-cache', codes[:5], cond, seed=42, finalize=False, streaming=True)
    a = s3gen._flow_noise(req, 100, torch.device(DEVICE), torch.float32)
    b = s3gen._flow_noise(req, 200, torch.device(DEVICE), torch.float32)
    assert a.data_ptr() == b.data_ptr()
    torch.testing.assert_close(a, b[:, :100], rtol=0, atol=0)
    s3gen.release_stream_state(['noise-cache'])
    assert 'noise-cache' not in s3gen._stream_noise
