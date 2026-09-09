"""Drive the SERVER's acoustic API in streaming mode and re-run the quality gate.

`eval_streaming_decode.py` proved the approach on a standalone prototype. This
proves the same properties through `ChatterboxS3Gen.decode()` -- the exact call
the acoustic stage makes -- so what ships is what was measured, including the
fixed noise bank, the pre-lookahead trim, the emit-once slice and the held-back
crossfade.
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

from eval_streaming_decode import seam_metric  # noqa: E402
from measure_prefix_stability import DEVICE, GOLDEN, MODEL_DIR, load_conditioning  # noqa: E402
from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K  # noqa: E402
from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import (  # noqa: E402
    AcousticRequest,
    ChatterboxS3Gen,
)

SR = 24000


def chunk_schedule(n: int, first: int, growth: float, max_block: int) -> list[int]:
    """Cumulative code counts at which a chunk is decoded."""
    points, k, block = [], first, first
    while k < n:
        points.append(k)
        block = min(int(block * growth), max_block) if growth > 1.0 else block
        k += block
    points.append(n)
    return points


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="ex01")
    ap.add_argument("--cases", default="en_plain,ar_plain,en_long")
    ap.add_argument("--first-block", type=int, default=K.ACOUSTIC_STREAM_FIRST_BLOCK)
    ap.add_argument("--growth", type=float, default=K.ACOUSTIC_STREAM_BLOCK_GROWTH)
    ap.add_argument("--seed", type=int, default=1235)
    ap.add_argument("--outdir", default="/workspace/port/artifacts/streaming_server")
    ap.add_argument("--asr", action="store_true")
    ap.add_argument("--cfm", type=int, default=0,
                    help="override acoustic CFM solver steps (0 = checkpoint default)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    s3gen = ChatterboxS3Gen(apply_watermark=False)
    s3gen.load_weights_from_dir(MODEL_DIR)
    s3gen = s3gen.to(DEVICE).eval()
    cond = load_conditioning(args.voice)
    if args.cfm:
        s3gen.cfm_timesteps = args.cfm

    report = {
        "voice": args.voice,
        "first_block": args.first_block,
        "growth": args.growth,
        "crossfade_samples": s3gen.stream_crossfade_samples,
        "cfm_timesteps": s3gen.cfm_timesteps,
        "cases": {},
    }

    for case in args.cases.split(","):
        path = GOLDEN / args.voice / f"{case}.npz"
        if not path.exists():
            continue
        codes = torch.from_numpy(
            np.load(path)["gen_ids_valid"]
        ).reshape(-1).to(DEVICE, torch.long)
        n = int(codes.shape[0])

        # One-shot, exactly as the non-streaming server does it.
        one = s3gen.decode(
            [AcousticRequest(f"{case}-one", codes, cond, seed=args.seed)]
        )[0].audio.cpu().numpy()

        # Streaming, exactly as the acoustic stage would call it.
        points = chunk_schedule(n, args.first_block, args.growth, K.ACOUSTIC_STREAM_MAX_BLOCK)
        rid = f"{case}-stream"
        pieces, tokens_decoded, first_at = [], 0, None
        for k in points:
            res = s3gen.decode([
                AcousticRequest(
                    rid, codes[:k], cond, seed=args.seed,
                    finalize=(k >= n), streaming=True,
                )
            ])[0]
            tokens_decoded += k
            if res.audio.numel():
                if first_at is None:
                    first_at = k
                pieces.append(res.audio.cpu().numpy())
        streamed = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

        assert rid not in s3gen._stream_state, "streaming state leaked after finalize"

        sf.write(outdir / f"{case}_oneshot.wav", one, SR)
        sf.write(outdir / f"{case}_streamed.wav", streamed, SR)

        entry = {
            "n_codes": n,
            "chunk_points": points,
            "n_chunks": len(points),
            "tokens_decoded": tokens_decoded,
            "cost_multiplier": round(tokens_decoded / max(n, 1), 2),
            "first_audio_after_codes": first_at,
            "oneshot_samples": int(len(one)),
            "streamed_samples": int(len(streamed)),
            "length_delta_samples": int(len(streamed) - len(one)),
            "seam_streamed": seam_metric(streamed),
            "seam_oneshot": seam_metric(one),
        }
        report["cases"][case] = entry
        print(
            f"  {case:<12} n={n:>4} chunks={len(points)} cost x{entry['cost_multiplier']:<5} "
            f"first audio after {first_at} codes  len delta {entry['length_delta_samples']:>6} "
            f"samples  seam {entry['seam_streamed']['ratio']} vs {entry['seam_oneshot']['ratio']}"
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
            text, language = meta["text"], meta.get("language", "en")
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
                f"  {case:<12} CER {entry['streamed_cer']} vs {entry['oneshot_cer']}   "
                f"spk-sim {entry['streamed_speaker_similarity']} vs "
                f"{entry['oneshot_speaker_similarity']}"
            )

    out = Path("/workspace/port/artifacts/streaming_server_eval.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
