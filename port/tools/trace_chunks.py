"""Record chunk arrival times for one request and locate the playback underflow."""
import base64, json, sys, time
from pathlib import Path
import requests

BASE = "http://127.0.0.1:18091"
SR, BPS = 24000, 2
BUF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
text = ("There are two ways to do this. You can either keep the current plan and add the "
        "extra seats one at a time, which bills at the standard rate, or you can move to "
        "the annual plan, which includes ten seats and works out cheaper.")
model = requests.get(BASE + "/v1/models", timeout=10).json()["data"][0]["id"]
ref = "data:audio/wav;base64," + base64.b64encode(Path("/workspace/refvoices/en_ex01.wav").read_bytes()).decode()
payload = dict(model=model, input=text, language="en", ref_audio=ref,
               response_format="pcm", stream=True, stream_format="audio", seed=42)
t0 = time.perf_counter(); rows = []; audio = 0.0
with requests.post(BASE + "/v1/audio/speech", json=payload, stream=True, timeout=300) as r:
    r.raise_for_status()
    for c in r.iter_content(chunk_size=None):
        if not c: continue
        t = time.perf_counter() - t0
        dur = len(c) / (SR * BPS)
        rows.append((t, dur))
        audio += dur
ttfa = rows[0][0]
# Cumulative underflow with a BUF-second initial buffer.
stall = 0.0; buffered = 0.0; out = []
for i, (t, dur) in enumerate(rows):
    play_elapsed = max(0.0, t - ttfa - BUF) - stall
    under = max(0.0, play_elapsed - buffered)
    stall += under
    buffered += dur
    out.append(dict(i=i, arrive_s=round(t, 3), chunk_audio_s=round(dur, 3),
                    cum_audio_s=round(buffered, 3), gap_s=round(under, 3)))
print(json.dumps(dict(label=sys.argv[1], ttfa_s=round(ttfa, 3), chunks=len(rows),
                      total_audio_s=round(audio, 2), total_wall_s=round(rows[-1][0], 2),
                      cumulative_stall_s=round(stall, 3)), indent=None))
for r_ in out[:12]: print("   ", json.dumps(r_))
