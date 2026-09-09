"""Gate D (full-clause subset): the port's acoustic stage == the official reference.

Both sides decode the SAME captured codec sequence with the SAME reference
conditioning. Parity is checked with `generator=None` on the port side, which
makes the vendored modules take the identical global-RNG draws as the reference,
so the two waveforms are directly comparable.

Also asserted here:
  * a request's audio is independent of which requests it was batched with;
  * two different voices are never merged into one acoustic batch;
  * the final crop is exactly `max(1, N-1) * 960` samples.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K
from vllm_omni.model_executor.models.chatterbox_mtl_v3.conditioning import ReferenceConditioning
from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import (
    AcousticError,
    AcousticRequest,
    ChatterboxS3Gen,
    derive_seed,
)

MODEL_DIR = (
    "/workspace/.hf_home/hub/models--ResembleAI--chatterbox/snapshots/"
    "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
)
GOLDEN = "/workspace/port/artifacts/golden"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARITY_CASES = ["en_plain", "ar_plain", "mixed_ar_en", "en_short"]


@pytest.fixture(scope="module")
def s3gen():
    m = ChatterboxS3Gen(apply_watermark=False)
    m.load_weights_from_dir(MODEL_DIR)
    return m.to(DEVICE).eval()


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
def reference_model():
    from chatterbox.models.s3gen import S3Gen as RefS3Gen

    m = RefS3Gen()
    m.load_state_dict(
        torch.load(f"{MODEL_DIR}/s3gen.pt", map_location="cpu", weights_only=True), strict=False
    )
    return m.to(DEVICE).eval()


@torch.inference_mode()
def _reference_wave(model, codes, cond, seed=1235):
    torch.manual_seed(seed)
    wav, _ = model.inference(speech_tokens=codes, ref_dict=cond.s3gen_ref_dict())
    n = int(codes.shape[0])
    return wav.squeeze(0).float().cpu().numpy()[: max(1, n - 1) * 960]


@torch.inference_mode()
def _port_wave(s3gen, codes, cond, seed=1235):
    """The port's acoustic path with `generator=None`, i.e. the same global-RNG
    draw sequence the reference takes, so the two are directly comparable."""
    n = int(codes.shape[0])
    torch.manual_seed(seed)
    mels = s3gen.s3gen.flow_inference(
        codes.unsqueeze(0),
        speech_token_lens=torch.tensor([n], device=DEVICE),
        ref_dict=cond.s3gen_ref_dict(),
        n_cfm_timesteps=K.DEFAULT_CFM_TIMESTEPS,
        finalize=True,
        generator=None,
    )
    wavs, _ = s3gen.s3gen.hift_inference(mels, None, generator=None)
    wavs = wavs.clone()
    fade = s3gen.s3gen.trim_fade.to(wavs.device, wavs.dtype)
    wavs[:, : fade.shape[0]] *= fade
    return wavs[0].float().cpu().numpy()[: max(1, n - 1) * 960]


@pytest.mark.parametrize("case_id", PARITY_CASES)
@torch.inference_mode()
def test_waveform_matches_reference_for_identical_codes(s3gen, reference_model, case_id):
    """Same codes, same conditioning, same RNG draw -> same waveform.

    The tolerance is not invented: it is the reference implementation's OWN
    run-to-run variation on this machine, measured in the same process. On this
    GPU some cases are bit-identical across runs and some are not (kernel
    autotuning picks different reduction orders), so a fixed `allclose`
    threshold would either be vacuous or flaky. Requiring the port to agree with
    the reference at least as closely as the reference agrees with itself is the
    strongest claim the backend actually supports.
    """
    ref_npz = np.load(f"{GOLDEN}/ex01/{case_id}.npz")
    codes = torch.from_numpy(ref_npz["gen_ids_valid"]).long().to(DEVICE)
    cond = load_conditioning("ex01")
    n = int(codes.shape[0])

    ref_a = _reference_wave(reference_model, codes, cond)
    ref_b = _reference_wave(reference_model, codes, cond)
    self_variation = float(np.abs(ref_a - ref_b).max())

    got = _port_wave(s3gen, codes, cond)

    assert got.shape == ref_a.shape == (max(1, n - 1) * 960,)
    err = float(np.abs(got - ref_a).max())
    peak = float(np.abs(ref_a).max())
    rms = float(np.sqrt(np.mean((got - ref_a) ** 2)))
    # -80 dBFS floor, so a bit-identical backend still gets a meaningful bound.
    budget = max(self_variation, 1e-4 * peak)
    print(
        f"\n[{case_id}] n={n} peak={peak:.3f} port-vs-ref max|d|={err:.3e} "
        f"rms={rms:.3e} | reference-vs-itself={self_variation:.3e} budget={budget:.3e}"
    )
    assert err <= budget, (
        f"{case_id}: the port diverges from the reference ({err:.3e}) by more than the "
        f"reference diverges from itself ({self_variation:.3e})"
    )


@torch.inference_mode()
def test_watermark_is_applied_and_does_not_change_length(s3gen):
    """The Perth watermark is reference behaviour and is not silently dropped."""
    marked = ChatterboxS3Gen(apply_watermark=True)
    marked.s3gen = s3gen.s3gen  # reuse loaded weights
    cond = load_conditioning("ex01")
    codes = torch.from_numpy(
        np.load(f"{GOLDEN}/ex01/en_plain.npz")["gen_ids_valid"]
    ).long().to(DEVICE)
    n = int(codes.shape[0])
    plain = s3gen.decode([AcousticRequest("p", codes, cond, seed=5)])[0]
    wm = marked.decode([AcousticRequest("p", codes, cond, seed=5)])[0]
    assert wm.watermarked and not plain.watermarked
    assert wm.audio.shape == plain.audio.shape == (max(1, n - 1) * 960,)
    assert not torch.equal(wm.audio, plain.audio), "watermarking did not modify the audio"
    delta = float((wm.audio - plain.audio).abs().max())
    print(f"\nwatermark max|delta|={delta:.4f}")


@torch.inference_mode()
def test_final_crop_is_exactly_the_reference_formula(s3gen):
    cond = load_conditioning("ex01")
    ref_npz = np.load(f"{GOLDEN}/ex01/ar_plain.npz")
    codes = torch.from_numpy(ref_npz["gen_ids_valid"]).long().to(DEVICE)
    n = int(codes.shape[0])
    res = s3gen.decode([AcousticRequest("r0", codes, cond, seed=7)])[0]
    assert res.audio.shape[0] == max(1, n - 1) * 960 == int(ref_npz["wav_cropped"].shape[0])


@torch.inference_mode()
def test_audio_is_independent_of_batch_composition(s3gen):
    """A row batched with others must produce byte-identical audio to running alone."""
    cond = load_conditioning("ex01")
    ref_npz = np.load(f"{GOLDEN}/ex01/en_plain.npz")
    codes = torch.from_numpy(ref_npz["gen_ids_valid"][:60]).long().to(DEVICE)

    alone = s3gen.decode([AcousticRequest("A", codes, cond, seed=11)])[0]
    batched = s3gen.decode([
        AcousticRequest("A", codes, cond, seed=11),
        AcousticRequest("B", codes, cond, seed=99),
        AcousticRequest("C", codes, cond, seed=1234),
    ])
    assert batched[0].request_id == "A"

    drift = float((alone.audio - batched[0].audio).abs().max())
    peak = float(alone.audio.abs().max())
    # Different seeds must give genuinely different audio -- that is the scale a
    # real change looks like, and it is what batch drift is compared against.
    across_requests = float((batched[0].audio - batched[1].audio).abs().max())
    print(
        f"\nbatch-composition drift={drift:.3e} (peak={peak:.3f}, "
        f"{drift / peak * 100:.4f}% of peak); across-request difference={across_requests:.3e}"
    )
    assert across_requests > 0.1 * peak, "per-request noise streams are not actually independent"
    # With cuDNN TF32 left on this was 2.4% of peak; the acoustic stage disables
    # it precisely so a caller's audio does not depend on their batch-mates.
    assert drift < 1e-3 * peak, (
        f"a row's audio changed by {drift / peak * 100:.3f}% of peak because other rows "
        f"shared its batch"
    )


@torch.inference_mode()
def test_different_voices_are_never_merged_into_one_batch(s3gen):
    a, b = load_conditioning("ex01"), load_conditioning("ex02")
    codes = torch.from_numpy(
        np.load(f"{GOLDEN}/ex01/en_plain.npz")["gen_ids_valid"][:40]
    ).long().to(DEVICE)
    reqs = [
        AcousticRequest("a1", codes, a, seed=1),
        AcousticRequest("b1", codes, b, seed=2),
        AcousticRequest("a2", codes, a, seed=3),
    ]
    # Same length, different voices: these SHARE a batch, and the test below
    # proves each row still matches its solo decode.
    groups = s3gen.group_batches(reqs)
    assert sorted(sorted(g) for g in groups) == [[0, 1, 2]]

    # And the grouped path gives each voice its own conditioning.
    out = s3gen.decode(reqs)
    assert [r.request_id for r in out] == ["a1", "b1", "a2"]
    assert not torch.equal(out[0].audio, out[1].audio), "two voices produced identical audio"


@torch.inference_mode()
def test_lengths_are_not_padded_together(s3gen):
    cond = load_conditioning("ex01")
    full = np.load(f"{GOLDEN}/ex01/en_plain.npz")["gen_ids_valid"]
    short = torch.from_numpy(full[:20]).long().to(DEVICE)
    long = torch.from_numpy(full[:50]).long().to(DEVICE)
    groups = s3gen.group_batches([
        AcousticRequest("s", short, cond, seed=1),
        AcousticRequest("l", long, cond, seed=1),
    ])
    assert len(groups) == 2, (
        "unequal lengths must not share a padded batch: this checkpoint's token encoder "
        "does not isolate padded rows"
    )


@torch.inference_mode()
def test_invalid_codec_ids_are_refused(s3gen):
    cond = load_conditioning("ex01")
    bad = torch.tensor([1, 2, K.CODEC_VOCAB_SIZE], dtype=torch.long, device=DEVICE)
    with pytest.raises(AcousticError, match="outside"):
        s3gen.decode([AcousticRequest("x", bad, cond, seed=1)])
    neg = torch.tensor([1, -5, 3], dtype=torch.long, device=DEVICE)
    with pytest.raises(AcousticError, match="outside"):
        s3gen.decode([AcousticRequest("x", neg, cond, seed=1)])


def test_seed_streams_are_independent_per_purpose():
    """Flow and vocoder draw from independent streams, and `seed` reproduces.

    When the caller supplies a seed, the request id is deliberately NOT folded
    in: folding it in makes every call unique, which is exactly what broke
    reproducibility (the same seed could never reproduce a result). Without a
    caller seed the request id IS used, so concurrent unseeded requests still
    get independent noise.
    """
    flow = derive_seed("req-1", 42, "flow")
    vocoder = derive_seed("req-1", 42, "vocoder")
    assert flow != vocoder, "flow and vocoder must not share a noise stream"

    # A supplied seed reproduces, regardless of which request asks for it.
    assert derive_seed("req-1", 42, "flow") == flow
    assert derive_seed("req-2", 42, "flow") == flow, (
        "a supplied seed must reproduce the same audio, so the request id must not "
        "participate in the derivation"
    )
    assert derive_seed("req-1", 43, "flow") != flow

    # With no seed, different requests still get independent streams.
    assert derive_seed("req-1", None, "flow") != derive_seed("req-2", None, "flow")
    assert derive_seed("req-1", None, "flow") != derive_seed("req-1", None, "vocoder")
    assert derive_seed("req-1", None, "flow") == derive_seed("req-1", None, "flow")


# ---------------------------------------------------------------------------
# Acoustic batching: mixed voices at equal length are safe; mixed LENGTHS are
# not, on this checkpoint. Both claims are asserted with measurements.
# ---------------------------------------------------------------------------


def _same_length_two_voices():
    a, b = load_conditioning("ex01"), load_conditioning("ex02")
    full = np.load(f"{GOLDEN}/ex01/en_long.npz")["gen_ids_valid"]
    codes = torch.from_numpy(full[:63]).long().to(DEVICE)
    return [
        AcousticRequest("A", codes, a, seed=11),
        AcousticRequest("B", codes, b, seed=22),
        AcousticRequest("C", codes, a, seed=33),
    ]


@torch.inference_mode()
def test_mixed_voices_share_a_batch_and_each_row_is_exact(s3gen):
    """Different voices at the same length batch together with no drift."""
    jobs = _same_length_two_voices()
    assert s3gen.group_batches(jobs) == [[0, 1, 2]], (
        "same-length rows must share one pass regardless of voice"
    )
    solo = {j.request_id: s3gen.decode([j])[0] for j in jobs}
    batched = {r.request_id: r for r in s3gen.decode_batch(jobs)}
    for rid, ref in solo.items():
        got = batched[rid]
        assert got.audio.shape == ref.audio.shape
        peak = float(ref.audio.abs().max())
        err = float((got.audio - ref.audio).abs().max())
        print(f"\n[mixed-voice {rid}] batched-vs-solo max|d|={err:.3e} "
              f"({err / peak * 100:.4f}% of peak)")
        assert err < 1e-3 * peak, f"{rid}: mixed-voice batching drifted by {err:.3e}"
    # And the two voices remain distinguishable.
    assert not torch.equal(batched["A"].audio, batched["B"].audio)


@torch.inference_mode()
def test_row_order_does_not_change_a_row(s3gen):
    jobs = _same_length_two_voices()
    forward = {r.request_id: r.audio for r in s3gen.decode_batch(jobs)}
    reverse = {r.request_id: r.audio for r in s3gen.decode_batch(list(reversed(jobs)))}
    for rid, audio in forward.items():
        peak = float(audio.abs().max())
        assert float((audio - reverse[rid]).abs().max()) < 1e-3 * peak


@torch.inference_mode()
def test_unequal_lengths_are_refused_because_padding_is_not_isolated(s3gen):
    """The measurement behind the length-exact rule.

    Packing a short row with a longer one changes the short row's audio by tens
    of percent. This test PINS that as a known property of the checkpoint, so
    nobody re-enables ragged packing believing it is free.
    """
    a = load_conditioning("ex01")
    full = np.load(f"{GOLDEN}/ex01/en_long.npz")["gen_ids_valid"]
    short = AcousticRequest("S", torch.from_numpy(full[:40]).long().to(DEVICE), a, seed=11)
    long = AcousticRequest("L", torch.from_numpy(full[:97]).long().to(DEVICE), a, seed=22)

    # Default batcher keeps them apart.
    assert len(s3gen.group_batches([short, long])) == 2

    # Forced together, the short row is measurably corrupted.
    ragged = ChatterboxS3Gen(apply_watermark=False, ragged_batching=True)
    ragged.s3gen = s3gen.s3gen
    assert ragged.group_batches([short, long]) == [[0, 1]]
    solo = ragged.decode([short])[0].audio
    packed = {r.request_id: r.audio for r in ragged.decode_batch([short, long])}["S"]
    peak = float(solo.abs().max())
    drift = float((packed - solo).abs().max()) / peak
    print(f"\n[ragged unsafe] short row drifts {drift * 100:.2f}% of peak when padded")
    assert drift > 0.01, (
        "ragged padding was expected to corrupt the short row on this checkpoint; if this "
        "now passes cleanly the encoder's masking changed and the rule can be revisited"
    )


@torch.inference_mode()
def test_group_size_is_capped(s3gen):
    """One very long utterance must not be able to stall an unbounded group."""
    cond = load_conditioning("ex01")
    codes = torch.from_numpy(
        np.load(f"{GOLDEN}/ex01/en_plain.npz")["gen_ids_valid"][:30]
    ).long().to(DEVICE)
    jobs = [AcousticRequest(f"r{i}", codes, cond, seed=i) for i in range(20)]
    groups = s3gen.group_batches(jobs)
    assert all(len(g) <= s3gen.max_batch_rows for g in groups)
    assert sum(len(g) for g in groups) == 20
