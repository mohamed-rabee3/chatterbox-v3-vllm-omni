"""Acoustic decode cost and quality vs the reference prompt length and CFM steps.

The acoustic decode runs the flow over `prompt + k` tokens, and the reference
prompt is 250 codec tokens (10 s). Measured at block sizes 5..50 the decode
takes ~240 ms whatever k is -- the prompt IS the cost. It therefore sets both
the time-to-first-audio floor and the throughput ceiling, on every chunk.

Both levers here are quality changes, so each is scored, never just timed:
shortening the prompt is less evidence of the speaker, and cutting CFM
timesteps is a coarser ODE solve.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/workspace/repos/vllm-omni")
sys.path.insert(0, "/workspace/port/tools")

from measure_prefix_stability import DEVICE, GOLDEN, MODEL_DIR, load_conditioning
from vllm_omni.model_executor.models.chatterbox_mtl_v3.conditioning import ReferenceConditioning
from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import (
    AcousticRequest, ChatterboxS3Gen,
)

SR = 24000
RATIO = 2


def truncate_prompt(cond: ReferenceConditioning, n: int) -> ReferenceConditioning:
    """Keep the LAST n prompt codes -- the part adjacent to what comes next."""
    have = int(cond.prompt_token_len.reshape(-1)[0].item())
    n = min(n, have)
    tok = cond.prompt_token.reshape(1, -1)[:, have - n : have]
    feat = cond.prompt_feat.reshape(1, -1, 80)[:, (have - n) * RATIO : have * RATIO]
    return ReferenceConditioning(
        cache_key=f"{cond.cache_key}-p{n}",
        speaker_emb=cond.speaker_emb,
        cond_prompt_speech_tokens=cond.cond_prompt_speech_tokens,
        prompt_token=tok.contiguous(),
        prompt_token_len=torch.tensor([n], device=tok.device, dtype=torch.long),
        prompt_feat=feat.contiguous(),
        embedding=cond.embedding,
    )


def timed(fn, repeats=3):
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="ex01")
    ap.add_argument("--case", default="en_plain")
    ap.add_argument("--prompts", default="250,150,100,75,50")
    ap.add_argument("--cfm", default="10,6,4")
    ap.add_argument("--block", type=int, default=25)
    ap.add_argument("--outdir", default="/workspace/port/artifacts/prompt_cost")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    s3gen = ChatterboxS3Gen(apply_watermark=False)
    s3gen.load_weights_from_dir(MODEL_DIR)
    s3gen = s3gen.to(DEVICE).eval()
    base = load_conditioning(args.voice)
    cap = np.load(GOLDEN / args.voice / f"{args.case}.npz")
    codes = torch.from_numpy(cap["gen_ids_valid"]).reshape(-1).to(DEVICE, torch.long)
    manifest = json.loads((GOLDEN / args.voice / "manifest.json").read_text())
    meta = {c["case_id"]: c for c in manifest["cases"]}[args.case]
    text, language = meta["text"], meta.get("language", "en")

    default_cfm = s3gen.cfm_timesteps
    rows = []
    for pn in [int(x) for x in args.prompts.split(",")]:
        cond = truncate_prompt(base, pn)
        for cfm in [int(x) for x in args.cfm.split(",")]:
            s3gen.cfm_timesteps = cfm
            rid = f"p{pn}c{cfm}"
            def run(rid=rid, cond=cond):
                s3gen._stream_state.pop(rid, None)
                s3gen.decode([AcousticRequest(rid, codes[:args.block], cond,
                                              seed=1, finalize=False, streaming=True)])
            dt = timed(run)
            s3gen._stream_state.pop(rid, None)
            full = s3gen.decode([AcousticRequest(f"{rid}-full", codes, cond, seed=1)])[0]
            wav = full.audio.cpu().numpy()
            sf.write(outdir / f"{rid}.wav", wav, SR)
            rows.append({"prompt_codes": pn, "cfm_timesteps": cfm,
                         "first_chunk_ms": round(dt * 1000, 1), "wav": f"{rid}.wav"})
            print(f"  prompt={pn:>3} cfm={cfm:>2} -> first chunk {dt*1000:6.1f} ms")
    s3gen.cfm_timesteps = default_cfm

    from verify_audio import ASR, SpeakerSimilarity, cer, normalize_for_asr
    asr, spk = ASR(), SpeakerSimilarity()
    ra, rsr = sf.read(f"/workspace/refvoices/en_{args.voice}.wav", dtype="float32")
    print()
    for r in rows:
        wav, sr = sf.read(outdir / r["wav"], dtype="float32")
        hyp = asr.transcribe(wav, sr, language)
        r["cer"] = round(cer(normalize_for_asr(text, language),
                             normalize_for_asr(hyp, language)), 4)
        r["speaker_similarity"] = round(spk.similarity(wav, sr, ra, rsr), 4)
        print(f"  prompt={r['prompt_codes']:>3} cfm={r['cfm_timesteps']:>2} "
              f"{r['first_chunk_ms']:>6.1f} ms  CER={r['cer']:.4f}  spk={r['speaker_similarity']:.4f}")

    out = Path("/workspace/port/artifacts/prompt_cost.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
