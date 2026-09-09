# Streaming latency optimization — 2026-09-09

**Follow-up capacity qualification:** the playback-paced Locust sweep in
[`capacity-paced/README.md`](capacity-paced/README.md) measures 1/2/4/8/30
callers and records a newly exposed reproducible truncation failure. The zero
failures below describe the earlier workload only; they do not establish
production reliability. The follow-up also confirms substantial playback gaps
with 30 conversational callers.

Measurements use one RTX 5090, FP32, the official multilingual V3 checkpoint,
four acoustic flow steps and the existing 5/15/35/75/... prefix schedule.
All HTTP load tests use Locust 2.46.5. See `environment.json` for exact versions,
model revision, voice sources and request settings.

This instance initially contained only the overlay. The runtime and weights were
restored, and three public speech recordings were prepared for the reference
voices. These are **new recordings**, so older repository numbers are historical
context, not directly comparable baselines. The `baseline-20` and `baseline-30`
directories are the before measurements collected here.

## Validated results

The selected configuration enables FP32 estimator compilation plus exact-shape
encoder/flow graph replay. All times below are seconds. Each conversation test
ran for 180 seconds; the additional no-think-time test ran for 120 seconds. Every
run reached its stated peak in-flight request count.

| Run | TTFA p50 | TTFA p95 | TTFA p99 | Completed turns | Failed / unfinished | Playback stall p95 |
|---|---:|---:|---:|---:|---:|---:|
| Before: 20 users | 1.598 | 2.460 | 2.550 | 151 | 0 / 0 | 34.280 |
| Selected: 20 users | 0.583 | 0.879 | 1.048 | 272 | 0 / 0 | 8.951 |
| Before: 30 users | 2.529 | 3.663 | 4.085 | 163 | 0 / 0 | 91.405 |
| Selected: 30 users | 0.601 | 1.192 | 1.388 | 288 | 0 / 0 | 16.368 |
| Selected: 30 users, no think time | 0.629 | 1.396 | 7.685 | 194 | 0 / 0 | 22.968 |

Across these three selected runs there were **854 measured requests**, including
100 intentional interruptions, with zero failed or unfinished requests.
Conversation p95 TTFA fell by 64% at 20 users and 67% at 30 users. At 30 users,
completed-turn throughput rose from 0.843 to 1.551 turns/second.

This **does not meet a consistent 500 ms p95 target**. Sustained 30-stream playback
still exceeds the measured FP32 configuration's capacity: the no-think-time run
reached 8.121 seconds maximum TTFA, and substantial buffer starvation remains.
Zero HTTP failures and a short first PCM chunk do not establish real-time
playback capacity. These are localhost measurements with the specific workload
and warmed reference voices below, not production-network latency guarantees.

The five selected HTTP smoke outputs (English, Arabic, mixed Arabic/English,
short text, and incremental streaming) retain their lengths and sample rates.
Maximum difference from the before render is 0.000397 of full-scale amplitude;
RMS difference is at most 0.000017. See `compiled-audio-parity.json`. Compilation
introduces small floating-point differences; the waveforms are not bit-identical.
All **59 targeted regression checks passed with the selected four-step,
compiled, graph-enabled acoustic settings**, including stream lengths, seams,
noise isolation, chunk transport, CFG state, and full-admission scheduling.
This is not a rerun of every historical fixture or a large held-out language
quality corpus.

The earlier graph-only results remain in `optimized-20`, `optimized-30`, and
`sustained-30`. Their p95 TTFA was 0.970 s / 1.301 s / 1.475 s respectively. To
use that alternative, set `acoustic_compile_estimator: false` in the profile.
The graph-only smoke comparison is `final-audio-parity.json` (maximum one PCM16
quantization step).

## Measurement contract

Each run uses English and Arabic conversations, three reference voices, mixed
utterance lengths, 1.5–6 seconds of think time, seeded requests, and simulated
interruptions. Locust ramps at five users/second and runs for 180 seconds, then
allows outstanding requests up to 90 seconds to finish. Three complete requests
warm the voice cache before measurement. GPU graph shapes not exercised by that
warmup are captured during the measured run.

TTFA is wall time from POST until the **first nonempty PCM body chunk**, including
TTFA from requests that are subsequently interrupted or fail. HTTP response
headers and empty chunks do not count. First chunks contain about 70 ms of audio.
`peak_inflight_requests` checks actual overlapping requests; the conversation
user count does not promise that every user is synthesizing throughout the run.
`playback_stall_s_p95` is estimated client buffer starvation after first audio,
computed from chunk arrival times and PCM duration. It is not an audio-quality
score. `requests_unfinished`, `failures`, and `barge_ins` are separate counters.

## Changes

- Batch CFG processing, repetition penalties and position/embedding updates;
  avoid per-row GPU-to-CPU synchronization in the AR hot path.
- Transfer the reference conditioning to CPU once per request, and stop copying
  hidden states that the acoustic stage never consumes.
- Use PIECEWISE CUDA capture for the nested Llama backbone. Python request
  identity, CFG state and reference payloads remain outside that graph.
- Transfer codec blocks at the acoustic decode rungs, instead of waking the
  acoustic stage and copying the voice payload for every generated token.
  The old YAML said growth 4 but the acoustic code used growth 2. Both sides now
  explicitly use 2, preserving the previously rendered waveform.
- Accept the transport's `meta.resumable` field before strict model payload
  validation. Without this, resumed requests could crash the entire acoustic
  engine under concurrency.
- Remove the repeated `.item()` synchronization in acoustic attention masks
  while preserving the all-masked-row repair.
- Move legacy CPU position-bias parameters onto the acoustic execution device
  during weight loading, eliminating repeated per-layer transfers.
- Compile the FP32 flow estimator and warm it before the server becomes ready.
  The isolated cold compile took about 291 seconds; the next server startup
  warmed it from the local Inductor cache in about 53 seconds.
- Opt in to exact-shape acoustic encoder/flow CUDA graphs. All tensor inputs,
  including each request's conditioning and noise, are copied before replay.
  A shared memory pool and 16-entry cap bound caching. Arbitrary final lengths
  and larger shapes use eager execution. FP32, solver steps, watermarking,
  lookahead, crossfade and the vocoder are preserved.
- Separate acoustic admission (32 streams) from execution batching. First
  chunks run together; continuation work alternates fairly with first chunks.
  Cancellation, memory allocation and streaming state transitions use the
  pinned upstream scheduler implementation. Only waiting requests with available
  admission capacity enter priority selection, avoiding a full-capacity deadlock.

## Rejected experiments

- `optimized-eager-30`: exposed the resumable-metadata crash; failed run.
- `rejected-growth4-30`: changed acoustic prefix boundaries and waveform; rejected.
- `large-acoustic-batch-30`: higher throughput, poor TTFA due to output batching;
  interrupted exploratory run.
- `rejected-admission4-30`: reducing `max_num_seqs` blocked new streams outside
  the engine, causing much worse TTFA; interrupted exploratory run.
- `scheduler-only-30`: bounded mixed execution batches, no acoustic graphs.
- `flow-graphs-mixed-batch-30`: flow graphs, but first chunks still shared their
  output batch with expensive continuation jobs.

- `flow-graphs-priority-30`: lower initial TTFA, but a full-admission deadlock;
  30 unfinished requests. Rejected.
- `rejected-admission-selection-30`: diagnostic reproduction of that stall;
  deliberately interrupted to apply the eligibility fix.

These directories preserve negative evidence; they are not successful optimized
benchmarks. Raw logs and PCM/WAV outputs remain local and are git-ignored.

## Reproduce

Install the complete overlay, including `vllm_omni/core/sched`, into vLLM-Omni
commit `b3dd45874a750f7edfa39bb02262804228e5ff7b`, with vLLM 0.28.0. The custom
scheduler follows that commit's API and must be reviewed when upgrading it.

```bash
OMNI_ROOT=/workspace/repos/vllm-omni LOG=- port/tools/serve.sh \
  --deploy-config chatterbox_mtl_v3_low_latency.yaml
LOAD_TTS_SEED=42 LEVELS="20 30" DUR=3m \
  OUT=port/artifacts/retest port/tools/sweep_concurrency.sh
# Optional sustained synthesis (no conversation think time):
LOAD_TTS_SEED=42 LOAD_WAIT_MIN=0 LOAD_WAIT_MAX=0 LEVELS=30 DUR=3m \
  OUT=port/artifacts/sustained port/tools/sweep_concurrency.sh
```

Paths to the local voice WAVs are in `port/tools/locustfile_conversation.py`.
`BASE`, `MODEL`, `HOST`, `PORT`, `OMNI_ROOT`, `VENV`, `DEPLOY_CONFIG`, and output directories are
configurable in the launch/test scripts. Keep long-lived deployment under the
instance's supervisor; the server used here binds to `127.0.0.1:18091`.

## Targeted regression checks

Run tests from the **full runtime checkout**. Running `python -m pytest` from the
overlay root can import its incomplete namespace instead of upstream modules.
The golden-dependent streaming tests also require the captured `ex01/en_plain`
and `ex01/en_long` fixtures under `/workspace/port/artifacts/golden`.

```bash
cd /workspace/repos/vllm-omni
CHATTERBOX_TEST_LOW_LATENCY=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  /venv/main/bin/python -m pytest -q \
  /workspace/chatterbox-v3-vllm-omni/port/tests/test_gate_b_sampling.py \
  /workspace/chatterbox-v3-vllm-omni/port/tests/test_batched_hotpath.py \
  /workspace/chatterbox-v3-vllm-omni/port/tests/test_stream_integrity.py \
  /workspace/chatterbox-v3-vllm-omni/port/tests/test_chunk_transport.py \
  /workspace/chatterbox-v3-vllm-omni/port/tests/test_acoustic_scheduling.py \
  /workspace/chatterbox-v3-vllm-omni/port/tests/test_acoustic_graphs.py \
  /workspace/chatterbox-v3-vllm-omni/port/tests/test_gate_e_streaming.py
```

## Managed instance

The selected service is `chatterbox` under supervisor, with automatic startup
and restart on unexpected process exit. Its wrapper uses the checked-in
`chatterbox_mtl_v3_low_latency.yaml` and binds privately to `127.0.0.1:18091`.
Cold startup can take several minutes for compilation; graph shapes not covered
by earlier requests are still captured lazily. No external port was exposed.

```bash
supervisorctl status chatterbox
supervisorctl restart chatterbox
curl http://127.0.0.1:18091/health
```
