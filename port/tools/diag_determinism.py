"""Where does seed reproducibility break: the AR stage or the acoustic stage?

If two runs with the same seed differ in LENGTH, the codec sequence differs and
the divergence is in autoregressive sampling. If they are the same length but
differ in samples, the divergence is in the acoustic stage's noise.
"""
from __future__ import annotations

import base64, io, json
from pathlib import Path

import numpy as np, requests, soundfile as sf

BASE = "http://127.0.0.1:18091"
REF = "data:audio/wav;base64," + base64.b64encode(
    Path("/workspace/refvoices/en_ex01.wav").read_bytes()).decode()


def run(model_id, seed, text="Hello, this is the reference implementation speaking."):
    body = {"model": model_id, "input": text, "language": "en", "ref_audio": REF,
            "response_format": "wav", "seed": seed}
    r = requests.post(f"{BASE}/v1/audio/speech", json=body, timeout=600)
    r.raise_for_status()
    a, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    return a, sr


def main():
    model_id = requests.get(f"{BASE}/v1/models", timeout=10).json()["data"][0]["id"]
    runs = [run(model_id, 4242) for _ in range(4)]
    lens = [len(a) for a, _ in runs]
    print("same-seed lengths:", lens)
    base = runs[0][0]
    for i, (a, _) in enumerate(runs[1:], 1):
        if len(a) == len(base):
            d = float(np.abs(a - base).max())
            print(f"  run0 vs run{i}: same length, max|diff|={d:.3e} "
                  f"({'IDENTICAL' if d == 0 else 'acoustic/AR numeric drift'})")
        else:
            print(f"  run0 vs run{i}: LENGTH DIFFERS ({len(base)} vs {len(a)}) -> AR sampling diverged")
    other, _ = run(model_id, 99)
    print("different-seed length:", len(other))
    Path("/workspace/port/artifacts/determinism.json").write_text(json.dumps(
        {"same_seed_lengths": lens, "different_seed_length": len(other)}, indent=2))


if __name__ == "__main__":
    main()
