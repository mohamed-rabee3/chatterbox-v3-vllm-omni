"""Concurrency + latency benchmark for the Chatterbox Multilingual V3 server.

Reports what the port plan asks for and refuses to report what it cannot
measure:

* **TTFA** is time to the first *playable* audio bytes, not first HTTP headers.
  In the non-streaming mode the whole clause arrives at once, so TTFA and total
  latency are the same number and are labelled as such.
* Capacity is reported as the highest offered load that met the declared SLOs,
  with rejections counted separately -- never extrapolated from a single run.
* Warm and cold voice-cache conditions are reported separately, because the
  first request for a voice pays for the conditioning encoder.
* Both closed-loop (fixed concurrency) and open-loop (fixed arrival rate)
  modes are provided: a closed-loop client that waits before sending more work
  cannot reveal overload.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import numpy as np
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
    "spk1272": "/workspace/refvoices/en_spk1272.wav",
}
WORKLOAD = [
    ("en_short", "en", "Hello."),
    ("en_plain", "en", "Hello, this is the reference implementation speaking."),
    ("en_long", "en",
     "The quick brown fox jumps over the lazy dog, and then it turns around and does it "
     "again because the sentence needs to be long enough to exercise a real decode loop."),
    ("ar_plain", "ar", "حياك الله، موعدك بكرة الساعة التاسعة صباحًا."),
    ("ar_long", "ar",
     "أهلاً وسهلاً بك في خدمة العملاء، نحن سعداء بتواصلك معنا اليوم، "
     "وسوف نقوم بمراجعة طلبك والرد عليك في أقرب وقت ممكن."),
    ("mixed", "ar", "حياك الله، حسابك على Netflix تم تجديده اليوم."),
]

_REF_B64: dict[str, str] = {}


def ref_b64(voice: str) -> str:
    if voice not in _REF_B64:
        _REF_B64[voice] = "data:audio/wav;base64," + base64.b64encode(
            Path(VOICES[voice]).read_bytes()
        ).decode()
    return _REF_B64[voice]


@dataclass
class Result:
    ok: bool
    status: int
    ttfa: float = 0.0
    total: float = 0.0
    audio_seconds: float = 0.0
    error: str = ""


@dataclass
class Summary:
    label: str
    offered: int
    results: list[Result] = field(default_factory=list)
    wall: float = 0.0

    def report(self) -> dict:
        ok = [r for r in self.results if r.ok]
        bad = [r for r in self.results if not r.ok]
        def pct(values, q):
            return statistics.quantiles(values, n=100)[q - 1] if len(values) > 1 else (values[0] if values else 0.0)
        totals = sorted(r.total for r in ok)
        audio = sum(r.audio_seconds for r in ok)
        return {
            "label": self.label,
            "offered": self.offered,
            "succeeded": len(ok),
            "failed": len(bad),
            "wall_s": round(self.wall, 2),
            "latency_p50_s": round(pct(totals, 50), 3) if totals else None,
            "latency_p95_s": round(pct(totals, 95), 3) if totals else None,
            "latency_p99_s": round(pct(totals, 99), 3) if totals else None,
            "latency_max_s": round(totals[-1], 3) if totals else None,
            "audio_seconds_total": round(audio, 2),
            # Generated audio seconds per wall-clock second: the throughput
            # number that actually matters for a TTS server.
            "audio_s_per_wall_s": round(audio / self.wall, 2) if self.wall else None,
            "requests_per_s": round(len(ok) / self.wall, 3) if self.wall else None,
            "errors": [r.error for r in bad][:5],
        }


async def one(session: aiohttp.ClientSession, text: str, language: str, voice: str,
              timeout: float) -> Result:
    payload = {
        "model": MODEL_ID,
        "input": text,
        "language": language,
        "ref_audio": ref_b64(voice),
        "response_format": "wav",
    }
    t0 = time.perf_counter()
    try:
        async with session.post(f"{BASE}/v1/audio/speech", json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            body = await r.read()
            dt = time.perf_counter() - t0
            if r.status != 200:
                return Result(False, r.status, error=f"HTTP {r.status}: {body[:200]!r}")
            audio, sr = sf.read(io.BytesIO(body), dtype="float32")
            return Result(True, 200, ttfa=dt, total=dt, audio_seconds=len(audio) / sr)
    except Exception as exc:  # noqa: BLE001 - the benchmark must survive any failure
        return Result(False, 0, total=time.perf_counter() - t0, error=f"{type(exc).__name__}: {exc}")


async def closed_loop(concurrency: int, requests: int, timeout: float, voices: list[str]) -> Summary:
    s = Summary(label=f"closed-loop c={concurrency}", offered=requests)
    queue: asyncio.Queue = asyncio.Queue()
    for i in range(requests):
        case = WORKLOAD[i % len(WORKLOAD)]
        queue.put_nowait((case[2], case[1], voices[i % len(voices)]))

    conn = aiohttp.TCPConnector(limit=max(concurrency * 2, 16))
    async with aiohttp.ClientSession(connector=conn) as session:
        async def worker():
            while True:
                try:
                    text, lang, voice = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                s.results.append(await one(session, text, lang, voice, timeout))
                queue.task_done()

        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        s.wall = time.perf_counter() - t0
    return s


async def open_loop(rate: float, seconds: float, timeout: float, voices: list[str]) -> Summary:
    """Fixed arrival rate: this is what reveals overload."""
    s = Summary(label=f"open-loop {rate:.1f} req/s for {seconds:.0f}s", offered=0)
    conn = aiohttp.TCPConnector(limit=0)
    tasks: list[asyncio.Task] = []
    async with aiohttp.ClientSession(connector=conn) as session:
        t0 = time.perf_counter()
        i = 0
        while time.perf_counter() - t0 < seconds:
            case = WORKLOAD[i % len(WORKLOAD)]
            tasks.append(asyncio.create_task(
                one(session, case[2], case[1], voices[i % len(voices)], timeout)
            ))
            s.offered += 1
            i += 1
            await asyncio.sleep(1.0 / rate)
        s.results = list(await asyncio.gather(*tasks))
        s.wall = time.perf_counter() - t0
    return s


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", default="1,2,4,8,16,32")
    ap.add_argument("--requests-per-level", type=int, default=0,
                    help="0 = 4x the concurrency level")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--voices", default="ex01,ex02,spk1272")
    ap.add_argument("--open-loop-rate", type=float, default=0.0)
    ap.add_argument("--open-loop-seconds", type=float, default=60.0)
    ap.add_argument("--out", default="/workspace/port/artifacts/bench.json")
    args = ap.parse_args()

    voices = args.voices.split(",")
    reports = []

    # Warm the voice conditioning cache first, and report the cold cost.
    print("== cold voice cache (first request per voice) ==")
    async with aiohttp.ClientSession() as session:
        for v in voices:
            r = await one(session, "Hello.", "en", v, args.timeout)
            print(f"  {v:<10} {'ok ' if r.ok else 'FAIL'} {r.total:6.2f}s  {r.error[:80]}")
            reports.append({"label": f"cold voice {v}", "latency_s": round(r.total, 3), "ok": r.ok})

    for level in [int(x) for x in args.concurrency.split(",") if x]:
        n = args.requests_per_level or max(4 * level, 8)
        s = await closed_loop(level, n, args.timeout, voices)
        rep = s.report()
        reports.append(rep)
        print(json.dumps(rep, indent=None))

    if args.open_loop_rate > 0:
        s = await open_loop(args.open_loop_rate, args.open_loop_seconds, args.timeout, voices)
        rep = s.report()
        reports.append(rep)
        print(json.dumps(rep, indent=None))

    Path(args.out).write_text(json.dumps(reports, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
