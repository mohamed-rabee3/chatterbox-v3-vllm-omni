"""Quality gate for real (incremental) acoustic streaming.

The earlier gate asked whether a streamed prefix equals the one-shot decode. It
does not -- the token encoder is bidirectional, so the mel for codes 0..k moves
when codes after k arrive (measured at 5-36% of peak). But equality with the
one-shot decode is the wrong bar: a streamed utterance is allowed to be a
DIFFERENT yet equally valid rendering. What it may not have is a seam, a change
of voice, or lost words. So this measures:

* **Seams.** Streaming can never revise what it already sent, so the overlap is
  HELD BACK rather than retro-fixed: the last ``crossfade`` samples of each
  chunk are kept, and when the next chunk's decode arrives they are blended
  with its version of the same region before being emitted. Cost: one crossfade
  of extra latency (~10 ms), not a rewrite.
* **Intelligibility.** ASR CER of the streamed waveform against the prompt text,
  next to the one-shot decode's CER on the same text.
* **Identity.** Speaker-embedding cosine similarity against the reference clip,
  next to the one-shot decode's.
* **Cost.** Cumulative re-decode is O(n^2/B) token-units. Reported as a
  multiplier over the one-shot decode, because that is what it takes off
  throughput.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/workspace/repos/vllm-omni")
sys.path.insert(0, "/workspace/port/tools")

from measure_prefix_stability import (  # noqa: E402
    DEVICE,
    GOLDEN,
    MODEL_DIR,
    N_MELS,
    decode_prefix,
    load_conditioning,
)
from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import ChatterboxS3Gen  # noqa: E402

SR = 24000
SAMPLES_PER_MEL = 480      # 960 samples per code, 2 mel frames per code


@torch.inference_mode()
def vocode(s3gen: ChatterboxS3Gen, mel: torch.Tensor, seed: int) -> np.ndarray:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    wav, _ = s3gen.s3gen.hift_inference(mel, None, generator=g)
    wav = wav.clone()
    fade = s3gen.s3gen.trim_fade.to(wav.device, wav.dtype)
    wav[:, : fade.shape[0]] *= fade
    return wav[0].float().cpu().numpy()


@torch.inference_mode()
def stream_decode(
    s3gen: ChatterboxS3Gen,
    codes: torch.Tensor,
    cond,
    bank: torch.Tensor,
    block: int,
    crossfade: int,
    seed: int,
    growth: float = 1.0,
    max_block: int = 400,
) -> tuple[np.ndarray, dict]:
    """Emit-once streaming decode with a held-back crossfade.

    ``growth`` multiplies the block after each chunk. Cumulative re-decode costs
    O(n^2/B) with a fixed block, which is what makes long utterances expensive
    (x5.0 at 225 codes). Growing the block keeps the FIRST chunk small -- TTFA
    is set by the first block alone -- while later chunks cover more ground, so
    cost falls back toward linear. This is safe for playback because once audio
    is playing the client holds a buffer, and generation runs far faster than
    realtime, so the buffer grows faster than the chunks lengthen.
    """
    n = int(codes.shape[0])
    lookahead = s3gen.s3gen.flow.pre_lookahead_len

    out: list[np.ndarray] = []
    pending: np.ndarray | None = None
    emitted = 0                  # samples already emitted (excluding pending)
    tokens_decoded = 0
    first_audio_after_codes = None
    fade_in = np.linspace(0.0, 1.0, crossfade, dtype=np.float32)
    fade_out = 1.0 - fade_in

    k = block
    while True:
        k = min(k, n)
        finalize = k >= n
        mel = decode_prefix(s3gen, codes[:k], cond, bank, finalize=finalize)
        tokens_decoded += k
        wav = vocode(s3gen, mel, seed)
        # Samples this decode can legitimately supply.
        avail = min(len(wav), mel.shape[-1] * SAMPLES_PER_MEL)
        if finalize:
            avail = min(avail, max(1, n - 1) * 960)

        if avail > emitted + (crossfade if not finalize else 0):
            if pending is None:
                cut = avail - (crossfade if not finalize else 0)
                out.append(wav[:cut].copy())
                if first_audio_after_codes is None:
                    first_audio_after_codes = k
                pending = wav[cut:avail].copy() if not finalize else None
                emitted = cut
            else:
                blend = pending * fade_out + wav[emitted : emitted + crossfade] * fade_in
                out.append(blend.astype(np.float32))
                cut = avail - (crossfade if not finalize else 0)
                out.append(wav[emitted + crossfade : cut].copy())
                pending = wav[cut:avail].copy() if not finalize else None
                emitted = cut
        if finalize:
            break
        k += block
        block = min(int(block * growth), max_block) if growth > 1.0 else block

    if pending is not None:
        out.append(pending)
    audio = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
    return audio, {
        "chunks": len(out),
        "tokens_decoded": tokens_decoded,
        "cost_multiplier": round(tokens_decoded / max(n, 1), 2),
        "first_audio_after_codes": first_audio_after_codes,
        "ttfa_codes_fraction": round((first_audio_after_codes or n) / max(n, 1), 3),
    }


def seam_metric(audio: np.ndarray) -> dict:
    """Largest sample-to-sample jump, as a share of the signal's own scale.

    A click is a first-difference far outside the signal's normal range, so the
    ratio to the 99.99th percentile of |diff| is what matters, not the raw jump.
    """
    d = np.abs(np.diff(audio))
    if d.size == 0:
        return {"max_jump": 0.0, "p9999_jump": 0.0, "ratio": 0.0}
    p = float(np.percentile(d, 99.99))
    return {
        "max_jump": round(float(d.max()), 6),
        "p9999_jump": round(p, 6),
        "ratio": round(float(d.max()) / p, 2) if p > 0 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="ex01")
    ap.add_argument("--cases", default="en_plain,ar_plain,en_long")
    ap.add_argument("--block", type=int, default=25)
    ap.add_argument("--crossfade-ms", type=float, default=10.0)
    ap.add_argument("--growth", type=float, default=1.0,
                    help="multiply the block after each chunk (1.0 = fixed block)")
    ap.add_argument("--seed", type=int, default=1235)
    ap.add_argument("--outdir", default="/workspace/port/artifacts/streaming")
    ap.add_argument("--asr", action="store_true", help="run Whisper CER + speaker similarity")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    crossfade = int(SR * args.crossfade_ms / 1000.0)

    s3gen = ChatterboxS3Gen(apply_watermark=False)
    s3gen.load_weights_from_dir(MODEL_DIR)
    s3gen = s3gen.to(DEVICE).eval()
    cond = load_conditioning(args.voice)
    ratio = s3gen.s3gen.flow.token_mel_ratio
    declared = int(cond.prompt_token_len.reshape(-1)[0].item())

    report = {
        "voice": args.voice,
        "block_tokens": args.block,
        "growth": args.growth,
        "crossfade_ms": args.crossfade_ms,
        "cases": {},
    }

    for case in args.cases.split(","):
        path = GOLDEN / args.voice / f"{case}.npz"
        if not path.exists():
            continue
        cap = np.load(path)
        codes = torch.from_numpy(cap["gen_ids_valid"]).reshape(-1).to(DEVICE, torch.long)
        n = int(codes.shape[0])

        g = torch.Generator(device=DEVICE).manual_seed(args.seed)
        bank = torch.randn((1, N_MELS, (declared + n) * ratio), generator=g, device=DEVICE)

        one_mel = decode_prefix(s3gen, codes, cond, bank, finalize=True)
        one = vocode(s3gen, one_mel, args.seed)[: max(1, n - 1) * 960]

        streamed, stats = stream_decode(
            s3gen, codes, cond, bank, args.block, crossfade, args.seed, growth=args.growth
        )

        sf.write(outdir / f"{case}_oneshot.wav", one, SR)
        sf.write(outdir / f"{case}_streamed.wav", streamed, SR)

        entry = {
            "n_codes": n,
            "oneshot_samples": int(len(one)),
            "streamed_samples": int(len(streamed)),
            "length_delta_samples": int(len(streamed) - len(one)),
            "seam_streamed": seam_metric(streamed),
            "seam_oneshot": seam_metric(one),
            **stats,
        }
        report["cases"][case] = entry
        print(
            f"  {case:<12} n={n:>4}  cost x{stats['cost_multiplier']:<5} "
            f"first audio after {stats['first_audio_after_codes']} codes "
            f"({100*stats['ttfa_codes_fraction']:.0f}% of utterance)  "
            f"seam ratio streamed {entry['seam_streamed']['ratio']} vs "
            f"oneshot {entry['seam_oneshot']['ratio']}"
        )

    if args.asr:
        from verify_audio import ASR, SpeakerSimilarity, cer, normalize_for_asr

        manifest = json.loads((GOLDEN / args.voice / "manifest.json").read_text())
        texts = {c["case_id"]: c for c in manifest.get("cases", [])}
        asr, spk = ASR(), SpeakerSimilarity()
        ref_audio, ref_sr = sf.read(f"/workspace/refvoices/en_{args.voice}.wav", dtype="float32")
        for case, entry in report["cases"].items():
            meta = texts.get(case)
            if not meta:
                continue
            text, language = meta.get("text"), meta.get("language", "en")
            for kind in ("streamed", "oneshot"):
                audio, sr = sf.read(outdir / f"{case}_{kind}.wav", dtype="float32")
                hyp = asr.transcribe(audio, sr, language)
                entry[f"{kind}_cer"] = round(
                    cer(normalize_for_asr(text, language), normalize_for_asr(hyp, language)), 4
                )
                entry[f"{kind}_speaker_similarity"] = round(
                    spk.similarity(audio, sr, ref_audio, ref_sr), 4
                )
            print(
                f"  {case:<12} CER streamed {entry['streamed_cer']} vs oneshot "
                f"{entry['oneshot_cer']}   spk-sim streamed "
                f"{entry['streamed_speaker_similarity']} vs oneshot "
                f"{entry['oneshot_speaker_similarity']}"
            )

    out = Path("/workspace/port/artifacts/streaming_eval.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
