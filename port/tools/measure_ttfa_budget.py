"""Where the time before the first audible sample actually goes.

TTFA = (stage 0 generates the first B codes) + (stage 1 decodes them).
Both halves are measured here so the first-chunk size can be chosen from data
rather than guessed. The acoustic half is the surprising one: every chunk
decode runs the flow over `prompt + k` tokens, and the reference prompt is 250
codec tokens (10 s), so a 5-code chunk costs almost exactly what a 25-code
chunk costs. Shrinking the first block only pays off until the prompt
dominates -- this finds that point.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/repos/vllm-omni")
sys.path.insert(0, "/workspace/port/tools")

from measure_prefix_stability import DEVICE, GOLDEN, MODEL_DIR, load_conditioning
from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import (
    AcousticRequest, ChatterboxS3Gen,
)


def timed(fn, repeats=3):
    fn()  # warm
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    best = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best.append(time.perf_counter() - t0)
    return min(best)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="ex01")
    ap.add_argument("--case", default="en_long")
    ap.add_argument("--blocks", default="3,5,8,12,25,50")
    ap.add_argument("--cfm-timesteps", default="")
    ap.add_argument("--out", default="/workspace/port/artifacts/ttfa_budget.json")
    args = ap.parse_args()

    s3gen = ChatterboxS3Gen(apply_watermark=False)
    s3gen.load_weights_from_dir(MODEL_DIR)
    s3gen = s3gen.to(DEVICE).eval()
    cond = load_conditioning(args.voice)
    codes = torch.from_numpy(
        np.load(GOLDEN / args.voice / f"{args.case}.npz")["gen_ids_valid"]
    ).reshape(-1).to(DEVICE, torch.long)

    prompt_len = int(cond.prompt_token_len.reshape(-1)[0].item())
    report = {"voice": args.voice, "prompt_codes": prompt_len,
              "cfm_timesteps": s3gen.cfm_timesteps, "blocks": {}}
    print(f"acoustic prompt = {prompt_len} codes ({prompt_len/25:.1f}s of reference)")

    for b in [int(x) for x in args.blocks.split(",")]:
        if b > int(codes.numel()):
            continue
        rid = f"bench-{b}"
        def run(b=b, rid=rid):
            s3gen._stream_state.pop(rid, None)
            s3gen.decode([AcousticRequest(rid, codes[:b], cond, seed=1,
                                          finalize=False, streaming=True)])
        dt = timed(run)
        s3gen._stream_state.pop(rid, None)
        emitted = max(0, b - 3) * 0.04
        report["blocks"][b] = {
            "decode_s": round(dt, 4),
            "audio_emitted_s": round(emitted, 3),
            "rtf": round(emitted / dt, 2) if dt else None,
        }
        print(f"  block={b:>3} codes -> decode {dt*1000:7.1f} ms, "
              f"emits {emitted:4.2f}s audio (rtf {emitted/dt:5.2f}x)")

    if args.cfm_timesteps:
        print("\ncfm timesteps sweep at block=8:")
        report["cfm_sweep"] = {}
        for n in [int(x) for x in args.cfm_timesteps.split(",")]:
            s3gen.cfm_timesteps = n
            rid = f"cfm-{n}"
            def run(rid=rid):
                s3gen._stream_state.pop(rid, None)
                s3gen.decode([AcousticRequest(rid, codes[:8], cond, seed=1,
                                              finalize=False, streaming=True)])
            dt = timed(run)
            s3gen._stream_state.pop(rid, None)
            report["cfm_sweep"][n] = round(dt, 4)
            print(f"  cfm_timesteps={n:>3} -> {dt*1000:7.1f} ms")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
