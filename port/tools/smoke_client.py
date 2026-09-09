"""End-to-end HTTP smoke test for the Chatterbox Multilingual V3 server.

Exercises the real serving path: EN / AR / mixed text, two different reference
voices, the non-streaming and streaming transports, and the failure paths that
must NOT return audio.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

BASE = "http://127.0.0.1:18091"


def served_model_id() -> str:
    """Ask the server what it calls itself, rather than guessing."""
    import requests as _rq

    data = _rq.get(f"{BASE}/v1/models", timeout=10).json()["data"]
    return data[0]["id"]


MODEL_ID = served_model_id()
VOICES = {
    "ex01": "/workspace/refvoices/en_ex01.wav",
    "ex02": "/workspace/refvoices/en_ex02.wav",
}
CASES = [
    ("en_plain", "en", "Hello, this is the reference implementation speaking.", "ex01"),
    ("ar_plain", "ar", "حياك الله، موعدك بكرة الساعة التاسعة صباحًا.", "ex01"),
    ("mixed_ar_en", "ar", "حياك الله، حسابك على Netflix تم تجديده اليوم.", "ex02"),
    ("en_short", "en", "Hello.", "ex02"),
]


def speech(payload: dict, timeout: float = 240.0) -> requests.Response:
    return requests.post(f"{BASE}/v1/audio/speech", json=payload, timeout=timeout)


def request_payload(text: str, language: str, voice: str, **kw) -> dict:
    wav = Path(VOICES[voice]).read_bytes()
    body = {
        "model": MODEL_ID,
        "input": text,
        "language": language,
        "ref_audio": "data:audio/wav;base64," + base64.b64encode(wav).decode(),
        "response_format": "wav",
        "seed": 1234,
    }
    body.update(kw)
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/port/artifacts/http")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []

    print("== non-streaming ==")
    for case_id, language, text, voice in CASES:
        t0 = time.perf_counter()
        r = speech(request_payload(text, language, voice))
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            failures.append(f"{case_id}: HTTP {r.status_code} {r.text[:300]}")
            print(f"  {case_id:<14} FAIL {r.status_code} {r.text[:200]}")
            continue
        path = out / f"{case_id}.wav"
        path.write_bytes(r.content)
        audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        rms = float(np.sqrt(np.mean(audio**2)))
        print(
            f"  {case_id:<14} ok  {len(audio)/sr:6.2f}s @{sr}Hz  rms={rms:.4f}  "
            f"latency={dt:.2f}s -> {path.name}"
        )
        if len(audio) == 0 or rms < 1e-4:
            failures.append(f"{case_id}: produced silence (rms={rms:.2e})")

    print("\n== rejected requests (must NOT return audio) ==")
    bad = [
        ("empty text", {"input": "   "}),
        ("no ref_audio", {"ref_audio": None}),
        ("unknown language", {"language": "klingon"}),
        ("unqualified language", {"language": "es"}),
        ("speed unsupported", {"speed": 1.5}),
        ("bad extra param", {"extra_params": {"nonsense": 1}}),
        ("cfg out of range", {"extra_params": {"cfg_weight": 99}}),
    ]
    for label, override in bad:
        payload = request_payload("Hello there.", "en", "ex01")
        payload.update(override)
        r = speech(payload, timeout=60)
        ok = r.status_code >= 400
        print(f"  {label:<22} -> HTTP {r.status_code} {'ok' if ok else 'UNEXPECTEDLY ACCEPTED'}")
        if not ok:
            failures.append(f"{label}: accepted (HTTP {r.status_code})")

    print("\n== streaming (raw audio) ==")
    payload = request_payload(CASES[0][2], "en", "ex01", stream=True, stream_format="audio",
                              response_format="pcm")
    t0 = time.perf_counter()
    chunks, first_at = [], None
    with requests.post(f"{BASE}/v1/audio/speech", json=payload, stream=True, timeout=240) as r:
        if r.status_code != 200:
            failures.append(f"stream: HTTP {r.status_code} {r.text[:300]}")
            print(f"  FAIL {r.status_code} {r.text[:200]}")
        else:
            for chunk in r.iter_content(chunk_size=None):
                if not chunk:
                    continue
                if first_at is None:
                    first_at = time.perf_counter() - t0
                chunks.append(chunk)
    if chunks:
        pcm = b"".join(chunks)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(out / "stream.wav", audio, 24000)
        print(
            f"  {len(chunks)} chunks, {len(audio)/24000:.2f}s audio, "
            f"first chunk at {first_at:.3f}s, total {time.perf_counter()-t0:.2f}s"
        )
        if len(audio) == 0:
            failures.append("stream: no audio")

    print("\n== summary ==")
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("  all smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
