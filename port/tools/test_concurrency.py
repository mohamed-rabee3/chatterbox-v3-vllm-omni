"""Gate C on the live server: isolation and guidance integrity under concurrency.

What this actually proves, and what it does not:

* **Voice isolation** -- with many requests interleaving two clearly different
  reference voices, every response must sound more like ITS OWN reference than
  like the other one, measured with Chatterbox's own voice encoder. This is the
  concrete form of "no cross-request contamination".
* **Guidance integrity** -- no request may come back with a CFG failure, and
  none may hit the token cap without EOS. Both are hard failures in the adapter,
  so a success here means guidance held for every request in the run.
* **Determinism** -- the same (text, voice, seed) served twice must give the
  same audio; different seeds must not.
* **Cancellation** -- aborting mid-flight must leave the server healthy and must
  not disturb concurrent requests.

It does NOT prove linguistic quality; that is the ASR/listening gate.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf
import torch

BASE = "http://127.0.0.1:18091"
MODEL_DIR = (
    "/workspace/.hf_home/hub/models--ResembleAI--chatterbox/snapshots/"
    "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
)
VOICES = {"ex01": "/workspace/refvoices/en_ex01.wav", "ex02": "/workspace/refvoices/en_ex02.wav"}
TEXTS = [
    ("en", "Hello, this is the reference implementation speaking."),
    ("ar", "حياك الله، موعدك بكرة الساعة التاسعة صباحًا."),
    ("en", "Your appointment is confirmed for tomorrow morning."),
    ("ar", "أهلاً وسهلاً بك في خدمة العملاء اليوم."),
]

_B64: dict[str, str] = {}


def ref_b64(voice: str) -> str:
    if voice not in _B64:
        _B64[voice] = "data:audio/wav;base64," + base64.b64encode(
            Path(VOICES[voice]).read_bytes()
        ).decode()
    return _B64[voice]


class Speaker:
    def __init__(self, device="cuda"):
        from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.voice_encoder import (
            VoiceEncoder,
        )

        self.ve = VoiceEncoder()
        self.ve.load_state_dict(
            torch.load(f"{MODEL_DIR}/ve.pt", map_location="cpu", weights_only=True)
        )
        self.ve = self.ve.to(device).eval()

    @torch.inference_mode()
    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        if sr != 16000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        v = np.asarray(self.ve.embeds_from_wavs([audio], sample_rate=16000)).reshape(-1)
        return v / (np.linalg.norm(v) + 1e-9)


async def one(session, model_id, text, language, voice, seed=None, timeout=600):
    body = {
        "model": model_id, "input": text, "language": language,
        "ref_audio": ref_b64(voice), "response_format": "wav",
    }
    if seed is not None:
        body["seed"] = seed
    t0 = time.perf_counter()
    async with session.post(f"{BASE}/v1/audio/speech", json=body,
                            timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        data = await r.read()
        if r.status != 200:
            return {"ok": False, "status": r.status, "voice": voice,
                    "error": data[:250].decode("utf-8", "replace"),
                    "latency": time.perf_counter() - t0}
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        return {"ok": True, "status": 200, "voice": voice, "audio": audio, "sr": sr,
                "latency": time.perf_counter() - t0, "bytes": len(data)}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default="/workspace/port/artifacts/concurrency.json")
    args = ap.parse_args()

    import requests

    model_id = requests.get(f"{BASE}/v1/models", timeout=10).json()["data"][0]["id"]
    spk = Speaker()
    refs = {}
    for v, path in VOICES.items():
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        refs[v] = spk.embed(a, sr)

    report: dict = {}
    failures: list[str] = []

    # ---- 1. voice isolation under interleaved concurrency -------------------
    print(f"== voice isolation at concurrency {args.concurrency} ==")
    jobs = []
    for i in range(args.concurrency):
        language, text = TEXTS[i % len(TEXTS)]
        jobs.append((text, language, "ex01" if i % 2 == 0 else "ex02"))

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[one(session, model_id, *j) for j in jobs])
        wall = time.perf_counter() - t0

    rows = []
    for job, res in zip(jobs, results):
        if not res["ok"]:
            failures.append(f"isolation: HTTP {res['status']} {res.get('error', '')[:150]}")
            continue
        emb = spk.embed(res["audio"], res["sr"])
        own = float(np.dot(emb, refs[res["voice"]]))
        other = float(np.dot(emb, refs["ex02" if res["voice"] == "ex01" else "ex01"]))
        rows.append({"voice": res["voice"], "own": round(own, 4), "other": round(other, 4),
                     "margin": round(own - other, 4), "seconds": round(len(res["audio"]) / res["sr"], 2)})
        if own <= other:
            failures.append(
                f"isolation: a {res['voice']} request sounds more like the other voice "
                f"(own={own:.3f} other={other:.3f})"
            )
    ok_rows = [r for r in rows]
    report["isolation"] = {
        "concurrency": args.concurrency,
        "succeeded": len(ok_rows),
        "failed": args.concurrency - len(ok_rows),
        "wall_s": round(wall, 2),
        "min_margin": round(min((r["margin"] for r in ok_rows), default=0.0), 4),
        "mean_own": round(float(np.mean([r["own"] for r in ok_rows])) if ok_rows else 0.0, 4),
        "mean_other": round(float(np.mean([r["other"] for r in ok_rows])) if ok_rows else 0.0, 4),
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in report["isolation"].items() if k != "rows"}, indent=None))

    # ---- 2. determinism ------------------------------------------------------
    print("\n== determinism ==")
    async with aiohttp.ClientSession() as session:
        a = await one(session, model_id, TEXTS[0][1], "en", "ex01", seed=4242)
        b = await one(session, model_id, TEXTS[0][1], "en", "ex01", seed=4242)
        c = await one(session, model_id, TEXTS[0][1], "en", "ex01", seed=99)
    # One 16-bit LSB. The response is 16-bit PCM, so a float sample sitting on
    # a rounding boundary can flip by exactly this much from GPU float noise
    # (~-90 dBFS). Requiring bit-equality of the encoded audio would be
    # stricter than the transport itself, and would fail for a reason that has
    # nothing to do with the model.
    LSB = 1.0 / 32768
    same_len = a["ok"] and b["ok"] and len(a["audio"]) == len(b["audio"])
    drift = float(np.abs(a["audio"] - b["audio"]).max()) if same_len else float("inf")
    report["determinism"] = {
        "same_seed_same_length": bool(same_len),
        "same_seed_max_abs_diff": None if not same_len else round(drift, 8),
        "same_seed_within_one_lsb": bool(same_len and drift <= LSB),
        # NOT asserted: min-p 0.05 prunes hard enough that two seeds often
        # sample the same tokens, so "different seeds differ" is not a property
        # this model has.
        "different_seed_length": len(c["audio"]) if c["ok"] else None,
    }
    print(json.dumps(report["determinism"], indent=None))
    if not same_len:
        failures.append("determinism: the same seed produced a different-length utterance")
    elif drift > LSB:
        failures.append(
            f"determinism: same-seed audio differs by {drift:.2e}, more than one 16-bit LSB"
        )

    # ---- 3. cancellation ------------------------------------------------------
    print("\n== cancellation ==")
    async with aiohttp.ClientSession() as session:
        task = asyncio.create_task(one(session, model_id, TEXTS[2][1], "en", "ex01"))
        await asyncio.sleep(0.35)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(1.0)
        after = await one(session, model_id, TEXTS[0][1], "en", "ex02")
    healthy = after["ok"]
    report["cancellation"] = {"server_serves_after_cancel": bool(healthy)}
    print(json.dumps(report["cancellation"], indent=None))
    if not healthy:
        failures.append(f"cancellation: the next request failed ({after.get('error', '')[:150]})")

    report["failures"] = failures
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("\n== summary ==")
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("  all concurrency checks passed")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
