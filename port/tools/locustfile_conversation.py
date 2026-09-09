"""Conversational load profile for the Chatterbox Multilingual V3 vLLM-Omni server.

This models N concurrent *conversations* with a live voice agent, not N isolated
TTS calls:

* A user holds a session. It keeps one agent voice for the whole call, so the
  conditioning cache is warm after the first turn -- which is what a real
  deployment looks like, and what makes the numbers comparable across turns.
* A caller speaks one language for the whole call. Code-switched turns (Arabic
  carrying English brand names) appear inside the Arabic calls, because that is
  how the real traffic looks, not as a separate synthetic language.
* Turns are mixed by length the way agent speech actually is: mostly short
  acknowledgements and medium answers, occasionally a long explanation.
* Between turns the user "thinks" -- the human is speaking and the ASR/LLM
  stages upstream are running. Without that gap this would be a throughput
  benchmark, not a conversation.
* A small share of turns are barged in on: the caller interrupts and the client
  drops the stream. These are counted separately and are NOT failures -- they
  are the single most common real event a live agent must survive.

TTFA measures the first nonempty PCM chunk. Playback starvation is also reported:
a fast first chunk alone does not establish real-time streaming capacity. Set
LOAD_PACE_PLAYBACK=1 to wait for playback before the next conversational gap;
LOAD_PLAYBACK_BUFFER_S (default 0.2) adds a modeled initial client buffer. No
STT or LLM is called, and their latency is not included in the TTS measurements.

Run:
    locust -f locustfile_conversation.py --headless -u 30 -r 2 -t 6m \
           --host http://127.0.0.1:18091
"""

from __future__ import annotations

import base64
import itertools
import json
import os
import random
import statistics
import time
from pathlib import Path

import requests
from locust import User, between, events, task

SAMPLE_RATE = 24000          # server output rate for pcm
BYTES_PER_SAMPLE = 2         # int16
ARTIFACTS = Path(os.environ.get("LOAD_ARTIFACTS", "/workspace/port/artifacts"))

# Time-to-first-audio targets for a live voice agent. These are reported
# against, not enforced -- a request slower than the target is still a
# successful request, it is just a bad conversational experience.
TTFA_GOOD_S = 1.0
TTFA_TOLERABLE_S = 2.0

VOICES = {
    "ex01": "/workspace/refvoices/en_ex01.wav",
    "ex02": "/workspace/refvoices/en_ex02.wav",
    "spk1272": "/workspace/refvoices/en_spk1272.wav",
}

# ---------------------------------------------------------------------------
# Conversation material. Turn classes carry their own realistic length.
# ---------------------------------------------------------------------------

SCRIPT = {
    "en": {
        "greet": [
            "Hello, thanks for calling support. How can I help you today?",
            "Good morning, you have reached the customer care line. What can I do for you?",
        ],
        "ack": [
            "Sure, one moment.",
            "Got it.",
            "Okay, let me check that.",
            "Of course.",
            "Right, I see.",
        ],
        "reply": [
            "I can see your account here, and the last payment went through on the fourth of this month.",
            "That order shipped yesterday evening, so it should reach you within two working days.",
            "I have updated the address on file. You will get a confirmation message shortly.",
            "Your subscription renews automatically, but you can turn that off at any time.",
        ],
        "explain": [
            "So here is what happened. The first charge was an authorisation hold, which your bank "
            "places when the card is verified, and it drops off on its own after a few days. The "
            "second line is the actual payment for the month. You have not been billed twice, but I "
            "understand why it looks that way on the statement, and I can send you a written breakdown.",
            "There are two ways to do this. You can either keep the current plan and add the extra "
            "seats one at a time, which bills at the standard rate, or you can move to the annual "
            "plan, which includes ten seats and works out cheaper if you expect the team to grow. "
            "I can apply either one before the end of today's call.",
        ],
        "close": [
            "Is there anything else I can help you with today?",
            "Thanks for calling, and have a good day.",
        ],
    },
    "ar": {
        "greet": [
            "حياك الله، معك خدمة العملاء. كيف أقدر أساعدك اليوم؟",
            "أهلاً وسهلاً بك، تفضل كيف أخدمك؟",
        ],
        "ack": [
            "تمام، لحظة من فضلك.",
            "أكيد.",
            "طيب، خلني أتأكد.",
            "حاضر.",
        ],
        "reply": [
            "أشوف حسابك هنا، وآخر عملية دفع تمت في الرابع من هذا الشهر.",
            "طلبك تم شحنه أمس بالمساء، ويوصلك خلال يومين عمل بإذن الله.",
            "حدثت العنوان في الحساب، وبيوصلك تأكيد على جوالك بعد قليل.",
            "حسابك على Netflix تم تجديده اليوم، والمبلغ خصم من نفس البطاقة.",
        ],
        "explain": [
            "أوضح لك اللي صار بالضبط. المبلغ الأول هو مجرد حجز مؤقت يقوم به البنك للتأكد من "
            "البطاقة، ويرجع لحسابك تلقائياً خلال أيام. والمبلغ الثاني هو قيمة الاشتراك الفعلية "
            "لهذا الشهر. يعني ما تم خصم المبلغ مرتين، لكن أتفهم إنه يظهر بهذا الشكل في كشف "
            "الحساب، وأقدر أرسل لك تفصيل مكتوب على الإيميل.",
            "عندك خيارين. إما تكمل على الباقة الحالية وتضيف المستخدمين واحد واحد بالسعر العادي، "
            "أو تنتقل إلى الباقة السنوية اللي تشمل عشرة مستخدمين وتطلع أوفر لك إذا كنت تتوقع "
            "زيادة في الفريق. أقدر أطبق أي خيار منهم قبل ما تنتهي المكالمة.",
        ],
        "close": [
            "فيه شي ثاني أقدر أساعدك فيه؟",
            "شكراً لتواصلك معنا، ويومك سعيد.",
        ],
    },
}

# Weighted turn mix for the middle of a call.
TURN_MIX = ["ack"] * 35 + ["reply"] * 50 + ["explain"] * 15

BARGE_IN_RATE = float(os.environ.get("LOAD_BARGE_IN_RATE", "0.05")) #      # share of turns the caller interrupts
PACE_PLAYBACK = os.environ.get("LOAD_PACE_PLAYBACK", "0") == "1"
PLAYBACK_BUFFER_S = float(os.environ.get("LOAD_PLAYBACK_BUFFER_S", "0.2"))
USER_IDS = itertools.count()
CALL_TURNS = (6, 12)      # turns per call before the caller hangs up

_REF_B64: dict[str, str] = {}


def ref_b64(voice: str) -> str:
    """Base64 reference audio, cached per voice like a real client would."""
    if voice not in _REF_B64:
        _REF_B64[voice] = "data:audio/wav;base64," + base64.b64encode(
            Path(VOICES[voice]).read_bytes()
        ).decode()
    return _REF_B64[voice]


# ---------------------------------------------------------------------------
# Sample collection -- locust's own stats mix every metric into one aggregate
# row, so TTFA is collected here as well and summarised honestly at the end.
# ---------------------------------------------------------------------------

class Samples:
    def __init__(self) -> None:
        self.ttfa: list[float] = []
        self.total: list[float] = []
        self.audio_s: list[float] = []
        self.by_class: dict[str, list[float]] = {}
        self.barge_ins = 0
        self.failures: list[str] = []
        self.started = 0
        self.inflight = 0
        self.peak_inflight = 0
        self.playback_stall_s: list[float] = []
        self.first_chunk_s: list[float] = []
        self.records: list[dict] = []
        self.captured: set[tuple[str, str]] = set()
        self.t0 = 0.0
        self.t1 = 0.0

    def add(self, turn_class: str, ttfa: float, total: float, audio_s: float) -> None:
        self.total.append(total)
        self.audio_s.append(audio_s)


SAMPLES = Samples()

MODEL_ID = os.environ.get("MODEL_ID", "")


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (q / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class ConversationUser(User):
    """One live caller holding one conversation at a time."""

    # The gap between agent turns: the human is talking and the upstream
    # ASR/LLM stages are running.
    wait_time = between(float(os.environ.get("LOAD_WAIT_MIN", "1.5")),
                        float(os.environ.get("LOAD_WAIT_MAX", "6.0")))

    def on_start(self) -> None:
        self.session = requests.Session()
        self.rng = random.Random(int(os.environ.get("LOAD_SEED", "42")) + next(USER_IDS))
        self._new_call()

    def on_stop(self) -> None:
        self.session.close()

    def _new_call(self) -> None:
        self.voice = self.rng.choice(list(VOICES))
        self.language = self.rng.choice(["en", "en", "ar", "ar", "ar"])
        self.turns_left = self.rng.randint(*CALL_TURNS)
        self.turn_index = 0

    def _next_turn(self) -> tuple[str, str]:
        script = SCRIPT[self.language]
        if self.turn_index == 0:
            klass = "greet"
        elif self.turns_left <= 1:
            klass = "close"
        else:
            klass = self.rng.choice(TURN_MIX)
        return klass, self.rng.choice(script[klass])

    @task
    def agent_turn(self) -> None:
        if self.turns_left <= 0:
            self._new_call()
        turn_class, text = self._next_turn()
        self.turn_index += 1
        self.turns_left -= 1

        payload = {
            "model": MODEL_ID,
            "input": text,
            "language": self.language,
            "ref_audio": ref_b64(self.voice),
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
        }
        if os.environ.get("LOAD_TTS_SEED"):
            payload["seed"] = int(os.environ["LOAD_TTS_SEED"])
        barge_in = self.rng.random() < BARGE_IN_RATE
        url = f"{self.host}/v1/audio/speech"

        t0 = time.perf_counter()
        ttfa = None
        capture_key = (self.voice, self.language)
        capture = os.environ.get("LOAD_CAPTURE_AUDIO") == "1" and capture_key not in SAMPLES.captured
        captured_chunks = []
        audio_bytes = 0
        playback_stall = 0.0
        SAMPLES.started += 1
        SAMPLES.inflight += 1
        SAMPLES.peak_inflight = max(SAMPLES.peak_inflight, SAMPLES.inflight)
        try:
            with self.session.post(url, json=payload, stream=True, timeout=300) as r:
                if r.status_code != 200:
                    body = r.text[:200]
                    raise RuntimeError(f"HTTP {r.status_code}: {body}")
                for chunk in r.iter_content(chunk_size=None):
                    if not chunk:
                        continue
                    if ttfa is None:
                        ttfa = time.perf_counter() - t0
                        # Include first audio even if a later chunk fails or
                        # the user interrupts. Completion-only TTFA is biased.
                        SAMPLES.ttfa.append(ttfa)
                        SAMPLES.by_class.setdefault(turn_class, []).append(ttfa)
                        SAMPLES.first_chunk_s.append(len(chunk) / (SAMPLE_RATE * BYTES_PER_SAMPLE))
                        events.request.fire(
                            request_type="TTFA", name=turn_class,
                            response_time=ttfa * 1000.0,
                            response_length=len(chunk), exception=None,
                            context={},
                        )
                        if barge_in:
                            # Caller interrupted: drop the stream mid-flight.
                            SAMPLES.barge_ins += 1
                            events.request.fire(
                                request_type="BARGE_IN", name=turn_class,
                                response_time=ttfa * 1000.0, response_length=0,
                                exception=None, context={},
                            )
                            return
                    elapsed_playback = time.perf_counter() - t0 - ttfa
                    buffered_until = audio_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE) + playback_stall
                    playback_stall += max(0.0, elapsed_playback - buffered_until)
                    audio_bytes += len(chunk)
                    if capture:
                        captured_chunks.append(chunk)
        except Exception as exc:  # noqa: BLE001 - a load test must survive anything
            dt = (time.perf_counter() - t0) * 1000.0
            SAMPLES.failures.append(f"{type(exc).__name__}: {exc}")
            events.request.fire(
                request_type="STREAM_ERROR", name=turn_class, response_time=dt,
                response_length=0, exception=exc, context={},
            )
            return
        finally:
            SAMPLES.inflight -= 1

        total = time.perf_counter() - t0
        audio_s = audio_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        if ttfa is None or audio_s <= 0.0:
            exc = RuntimeError("no audio returned")
            SAMPLES.failures.append(str(exc))
            events.request.fire(
                request_type="AUDIO", name=turn_class, response_time=total * 1000.0,
                response_length=0, exception=exc, context={},
            )
            return

        if capture and capture_key not in SAMPLES.captured:
            SAMPLES.captured.add(capture_key)
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            stem = ARTIFACTS / f"capture_{capture_key[0]}_{capture_key[1]}"
            stem.with_suffix(".pcm").write_bytes(b"".join(captured_chunks))
            stem.with_suffix(".json").write_text(json.dumps({
                "voice": capture_key[0], "language": capture_key[1], "text": text,
                "sample_rate": SAMPLE_RATE, "dtype": "int16", "ttfa_s": ttfa,
            }, ensure_ascii=False, indent=2))
        SAMPLES.playback_stall_s.append(playback_stall)
        SAMPLES.add(turn_class, ttfa, total, audio_s)
        # The unbuffered cumulative stall equals the extra initial buffering
        # needed to play this observed chunk trace without any underflow.
        buffered_stall = max(0.0, playback_stall - PLAYBACK_BUFFER_S)
        SAMPLES.records.append({
            "class": turn_class, "language": self.language, "voice": self.voice,
            "ttfa_s": ttfa, "total_s": total, "audio_s": audio_s,
            "unbuffered_stall_s": playback_stall,
            "buffered_stall_s": buffered_stall,
            "playback_start_s": ttfa + PLAYBACK_BUFFER_S,
        })
        events.request.fire(
            request_type="AUDIO", name=turn_class, response_time=total * 1000.0,
            response_length=audio_bytes, exception=None, context={},
        )
        if PACE_PLAYBACK:
            # TTS often finishes before the listener finishes hearing it.
            # Start the next conversational gap only after playback ends.
            playback_end = t0 + ttfa + PLAYBACK_BUFFER_S + audio_s + buffered_stall
            time.sleep(max(0.0, playback_end - time.perf_counter()))


@events.test_start.add_listener
def _on_start(environment, **_kw) -> None:
    global MODEL_ID, SAMPLES, USER_IDS
    SAMPLES = Samples()
    USER_IDS = itertools.count()
    random.seed(int(os.environ.get("LOAD_SEED", "42")))
    if not MODEL_ID:
        host = environment.host or "http://127.0.0.1:18091"
        MODEL_ID = requests.get(f"{host}/v1/models", timeout=30).json()["data"][0]["id"]
    if os.environ.get("LOAD_WARMUP", "1") == "1":
        for voice in VOICES:
            response = requests.post(f"{environment.host}/v1/audio/speech", json={
                "model": MODEL_ID, "input": "Hello, how can I help you today?",
                "language": "en", "ref_audio": ref_b64(voice), "response_format": "pcm",
                "stream": True, "stream_format": "audio", "seed": 42,
            }, timeout=300)
            response.raise_for_status()
            if not response.content:
                raise RuntimeError(f"warmup returned no audio for {voice}")
    SAMPLES.t0 = time.perf_counter()
    print(f"[load] model={MODEL_ID} host={environment.host}")


@events.test_stop.add_listener
def _on_stop(environment, **_kw) -> None:
    SAMPLES.t1 = time.perf_counter()
    wall = max(SAMPLES.t1 - SAMPLES.t0, 1e-9)
    ttfa = SAMPLES.ttfa
    n = len(ttfa)
    report = {
        "users": environment.parsed_options.num_users,
        "requests_started": SAMPLES.started,
        "requests_unfinished": SAMPLES.started - len(SAMPLES.total) - SAMPLES.barge_ins - len(SAMPLES.failures),
        "peak_inflight_requests": SAMPLES.peak_inflight,
        "workload_seed": os.environ.get("LOAD_SEED", "42"),
        "tts_seed": os.environ.get("LOAD_TTS_SEED"),
        "wait_s": [os.environ.get("LOAD_WAIT_MIN", "1.5"), os.environ.get("LOAD_WAIT_MAX", "6.0")],
        "barge_in_rate": BARGE_IN_RATE,
        "workload_version": 2,
        "pace_playback": PACE_PLAYBACK,
        "playback_buffer_s": PLAYBACK_BUFFER_S,
        "wall_s": round(wall, 1),
        "turns_completed": len(SAMPLES.total),
        "first_audio_observed": n,
        "first_chunk_audio_s_p50": round(_pct(SAMPLES.first_chunk_s, 50), 3),
        "playback_stall_s_p95": round(_pct(SAMPLES.playback_stall_s, 95), 3),
        "buffered_playback_stall_s_p95": round(_pct([
            max(0.0, s - PLAYBACK_BUFFER_S) for s in SAMPLES.playback_stall_s], 95), 3),
        "buffered_turns_stalled_over_100ms_pct": round(100.0 * sum(
            s > PLAYBACK_BUFFER_S + 0.1 for s in SAMPLES.playback_stall_s
        ) / max(len(SAMPLES.playback_stall_s), 1), 2),
        "playback_start_s_p95": round(_pct(ttfa, 95) + PLAYBACK_BUFFER_S, 3),
        "barge_ins": SAMPLES.barge_ins,
        "failures": len(SAMPLES.failures),
        "failure_examples": SAMPLES.failures[:5],
        "ttfa_s": {
            "p50": round(_pct(ttfa, 50), 3),
            "p90": round(_pct(ttfa, 90), 3),
            "p95": round(_pct(ttfa, 95), 3),
            "p99": round(_pct(ttfa, 99), 3),
            "max": round(max(ttfa), 3) if ttfa else None,
            "mean": round(statistics.fmean(ttfa), 3) if ttfa else None,
        },
        "ttfa_under_1s_pct": round(100.0 * sum(t <= TTFA_GOOD_S for t in ttfa) / n, 1) if n else None,
        "ttfa_under_2s_pct": round(100.0 * sum(t <= TTFA_TOLERABLE_S for t in ttfa) / n, 1) if n else None,
        "ttfa_by_turn_class_s": {
            k: {"n": len(v), "p50": round(_pct(v, 50), 3), "p95": round(_pct(v, 95), 3)}
            for k, v in sorted(SAMPLES.by_class.items())
        },
        "total_latency_s": {
            "p50": round(_pct(SAMPLES.total, 50), 3),
            "p95": round(_pct(SAMPLES.total, 95), 3),
            "max": round(max(SAMPLES.total), 3) if SAMPLES.total else None,
        },
        "audio_seconds_total": round(sum(SAMPLES.audio_s), 1),
        "audio_s_per_wall_s": round(sum(SAMPLES.audio_s) / wall, 2),
        "turns_per_s": round(len(SAMPLES.total) / wall, 3),
        "realtime_factor_p50": (
            round(_pct([a / t for a, t in zip(SAMPLES.audio_s, SAMPLES.total) if t > 0], 50), 2)
            if SAMPLES.total else None
        ),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "locust_conversation.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (ARTIFACTS / "completed_turns.jsonl").write_text("".join(
        json.dumps(record) + "\n" for record in SAMPLES.records))
    print("\n===== conversational load summary =====")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {out}")
