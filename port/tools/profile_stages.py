"""Where does a request's GPU time actually go: AR decode or acoustic decode?

Optimising without this is guesswork. Measures the acoustic stage directly at
several batch sizes, and derives the AR stage's cost from the measured decode
rate, for a typical utterance length.
"""
from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np, torch

MODEL_DIR = ("/workspace/.hf_home/hub/models--ResembleAI--chatterbox/snapshots/"
             "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18")
GOLDEN = "/workspace/port/artifacts/golden"


def main():
    from vllm_omni.model_executor.models.chatterbox_mtl_v3.conditioning import ReferenceConditioning
    from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import (
        AcousticRequest, ChatterboxS3Gen,
    )

    dev = "cuda"
    m = ChatterboxS3Gen(apply_watermark=True)
    m.load_weights_from_dir(MODEL_DIR)
    m = m.to(dev).eval()

    ref = np.load(f"{GOLDEN}/ex01/conditioning.npz")
    t = {k: torch.from_numpy(ref[k]).to(dev) for k in ref.files}
    cond = ReferenceConditioning(
        cache_key="prof", speaker_emb=t["speaker_emb"],
        cond_prompt_speech_tokens=t["cond_prompt_speech_tokens"].long(),
        prompt_token=t["prompt_token"].long(), prompt_token_len=t["prompt_token_len"].long(),
        prompt_feat=t["prompt_feat"], embedding=t["embedding"])
    codes = torch.from_numpy(np.load(f"{GOLDEN}/ex01/en_plain.npz")["gen_ids_valid"]).long().to(dev)
    n = int(codes.shape[0])

    rows = []
    for batch in (1, 2, 4, 8):
        reqs = [AcousticRequest(f"r{i}", codes, cond, seed=i) for i in range(batch)]
        m.decode(reqs)  # warm
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            m.decode(reqs)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 3
        rows.append({"batch": batch, "codes": n, "seconds": round(dt, 4),
                     "per_row_s": round(dt / batch, 4),
                     "audio_s_per_wall_s": round(batch * n / 25.0 / dt, 2)})
        print(json.dumps(rows[-1]))

    # Watermark cost on its own.
    audio = m.decode([AcousticRequest("w", codes, cond, seed=1)])[0].audio.cpu().numpy()
    t0 = time.perf_counter()
    for _ in range(3):
        m.watermark(audio)
    wm = (time.perf_counter() - t0) / 3
    print(json.dumps({"watermark_s": round(wm, 4), "audio_s": round(len(audio) / 24000, 2)}))
    rows.append({"watermark_s": round(wm, 4)})
    Path("/workspace/port/artifacts/profile_stages.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
