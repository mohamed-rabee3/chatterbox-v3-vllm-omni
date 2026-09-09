# TTS-only conversational capacity — 2026-09-09

This is a new Locust workload version, measured against the existing compiled
FP32 service on one RTX 5090. No STT or LLM runs in these tests. TTS latency
starts at the HTTP POST and includes preprocessing, queueing, generation and
local HTTP delivery. No production-network latency is included.

The earlier 20/30-user results established concurrent request completion, but
did not establish smooth real-time playback. This sweep checks lower caller
counts and waits for modeled playback to finish before starting the next turn.

## Interpretation

At a fixed 200 ms initial playback buffer, **two callers are the highest tested
multi-caller level meeting the working latency/playback criteria** in this
short sweep. Four warmed callers have p95 TTFA 549 ms, but p95 cumulative
buffered playback gaps 251 ms. Eight callers have substantial playback gaps.
Three callers were not tested; two is not a proven exact capacity ceiling.

The playback-paced **30-caller** run has p95 TTFA **946 ms**, p95 playback start
**1.146 seconds**, and p95 cumulative buffered gaps **12.281 seconds**. Of its
279 completed responses, 98.2% have over 100 ms of cumulative buffered gaps.
It has three failed streams, 15 intentional interruptions, and zero unfinished
requests. Peak overlapping TTS requests is 29: 30 is the caller population,
not a claim that all 30 continuously synthesize. Measured completed-audio
throughput is 7.80 audio seconds per wall second.

Across all six runs there are 565 requests: 527 completions, 30 intentional
interruptions, eight failures, and zero unfinished requests. The server remains
healthy after the sweep. These are short capacity samples, not a long soak test.

**No level is production-qualified:** a reproducible early-generation failure
appeared even at one caller. The server reports two output speech tokens for a
48-token input, rejects it with its truncation guard, and the client sees a
prematurely ended stream. The known request is preserved in
`truncation-reproduction.json`, with server evidence in
`server-failure-evidence.txt`. Retries and alternate seeds were not used to hide
failures. This short, seeded sample does not estimate a production failure rate.

The practical performance evidence is therefore a small caller count, with
four being sensitive to buffer choice, rather than a validated 30-caller
production deployment. These limits describe this software/configuration and
workload, not an inherent hardware limit of the RTX 5090.

## Workload and measurement

- English and Arabic, three cached reference voices, greetings, short replies,
  medium replies, and occasional long explanations. Exact texts remain in
  `port/tools/locustfile_conversation.py`.
- Each caller waits for modeled playback to finish, then waits 1.5–6 seconds
  before sending another text. This gap models request cadence only; it is not
  included in measured TTS latency. Interrupted turns skip remaining playback.
- 200 ms fixed client buffer, 5% configured immediate-after-first-chunk
  interruptions, seed 42 for workload and TTS. First chunks contain about
  70 ms of audio. Actual interruption counts are in each JSON report.
- Version 2 gives each user a distinct RNG stream and seeds the wait-time RNG.
  Version 1 could give users spawned together the same workload seed. Version 2
  exercised a text/voice combination that exposed the truncation failure.
  Old and new workloads are not directly comparable controlled A/B results.
- Runs `u1/u2/u4/u8` use Locust's 2-minute run limit. `warm-repeat/u4` and
  `warm-repeat/u30` use 3 minutes. All ramp at five users/second and allow
  90 seconds to drain. Exact measured wall time and counts are in the JSONs;
  the Locust time limit includes the sequential three-voice warmup.
- The service stayed running. The first four/eight-user tests included new
  acoustic graph captures. The later four-user run reuses those graph shapes.
- TTFA includes requests that later fail or are interrupted. Playback metrics
  use completed responses only. A playback stall is cumulative client buffer
  starvation over a response, not necessarily one continuous silence.
- Working criteria: p95 playback start <= 1 second, p95 cumulative buffered
  playback stall <= 100 ms, zero failures and zero unfinished requests. These
  are explicit provisional criteria, not a user-specified production SLA.

See `results.md` for every run, including failures. `completed_turns.jsonl`
preserves the per-response measurements behind playback percentiles.

## Buffer sensitivity

`buffer-tradeoff.json` models different fixed initial buffers using the recorded
completed-response traces. On the warmed four-user traces, a 400 ms buffer gives
approximately 955 ms p95 playback start and 61 ms p95 cumulative gaps. A 500 ms
buffer gives approximately 1.055 seconds p95 playback start and zero p95 gaps.
These are post-hoc calculations, **not fresh load tests**: changing playback
pacing can change offered load. They also exclude failed/interrupted streams.

Increasing a client buffer trades startup latency for fewer underflows; it
does not increase model throughput. Zero p95 gaps does not mean every response
is gap-free.

## Further optimization candidates

These are code-based hypotheses to benchmark, not measured speedup claims:

1. Batch compatible continuation chunks in small groups and schedule them
   against playback deadlines. The current scheduler batches initial chunks,
   alternates initial/continuation work, and selects one continuation at a time.
   Large mixed batches previously hurt TTFA, so merely raising the batch limit
   is not validated as a fix.
2. Benchmark BF16 acoustic inference with sensitive operations kept in FP32,
   checking intelligibility, voice similarity, artifacts and streaming seams.
   Current results are FP32; no reduced-precision performance is claimed.
3. Reduce repeated acoustic prefix work. The current decoder regenerates the
   growing prefix and emits only its new slice. Safe incremental state reuse
   requires model-level work and waveform validation.
4. Profile and compile/batch the remaining per-row vocoder work, retain reusable
   conditioning on device, and prewarm supported graph shapes before readiness.
   Prewarming addresses capture spikes, not sustained throughput by itself.

The truncation failure also needs resolution before reliability qualification.

## Reproduce

From the overlay repository root, with the existing compiled service running:

```bash
LOAD_TTS_SEED=42 LOAD_SEED=42 LOAD_PACE_PLAYBACK=1 \
  LOAD_PLAYBACK_BUFFER_S=0.2 LEVELS='1 2 4 8' DUR=2m \
  OUT=port/artifacts/capacity-retest port/tools/sweep_concurrency.sh
LOAD_TTS_SEED=42 LOAD_SEED=42 LOAD_PACE_PLAYBACK=1 \
  LOAD_PLAYBACK_BUFFER_S=0.2 LEVELS='4 30' DUR=3m \
  OUT=port/artifacts/capacity-retest/warm-repeat port/tools/sweep_concurrency.sh
/venv/main/bin/python port/tools/summarize_capacity.py \
  port/artifacts/capacity-retest
```

The sweep retains failed runs and continues to the next level, then returns
nonzero if any Locust run failed. Do not interpret that exit as an incomplete
test without reading the request/failure/unfinished counters.
