# Real-time voice-agent optimization at 20 concurrent callers — 2026-09-09

Target: 20 concurrent conversations on one RTX 5090, lowest latency and highest
throughput, **without trading quality for speed**.

The starting point for this round was the `capacity` profile, which had reached
its numbers partly by lowering quality: FP16 flow estimator, three flow-matching
steps instead of four, and a 256-code context window. This round **puts all of
that back** — FP32 throughout, four flow steps, the reference Perth watermark on
every emitted chunk — and recovers the speed from serialization and scheduling
instead. The resulting profile is
[`chatterbox_mtl_v3_realtime.yaml`](../../../vllm_omni/deploy/chatterbox_mtl_v3_realtime.yaml).

## Results

Locust conversational workload v2, playback-paced, 3 minutes per level, 200 ms
modelled client buffer, seeded, 5% barge-ins, EN+AR, three reference voices.
`before` is `capacity-stack` (FP16 estimator, 3 flow steps); `after` is this
profile (FP32, 4 flow steps).

| callers | TTFA p95 before → after | buffered stall p95 before → after | turns >100 ms before → after | turns/s before → after |
|---|---|---|---|---|
| 16 | 0.656 → 0.822 s | 0.690 → **0.417 s** | 29.9% → **23.5%** | 1.37 → 1.34 |
| **20** | 0.490 → **0.795 s** | 0.700 → **0.491 s** | 31.7% → **30.6%** | 1.68 → **1.70** |
| 24 | 0.585 → 0.843 s | 1.559 → **0.831 s** | 60.1% → **43.1%** | 1.94 → **2.08** |
| 30 | 0.604 → 0.921 s | 2.637 → **1.639 s** | 95.2% → **77.5%** | 2.22 → **2.51** |

Zero failures and zero unfinished requests at every level, 1500 turns total.
"Buffered stall" is cumulative playback underflow after a 200 ms client buffer.
The `after` sweep was started against a **freshly restarted** server, so it
includes the cold-start cost the profile will pay in production.

Playback stall falls 30-47% at every level, and throughput is equal or better
(+7% at 24 callers, +13% at 30) — delivered **at FP32 with four flow-matching
steps instead of FP16 with three**. TTFA rises (0.490 → 0.795 s p95 at 20
callers) because the first chunk was deliberately enlarged to stop the client
starving; it stays under 1 s p95 at every level tested.

**This is not "smooth for 20 users" by a strict reading.** 30.6% of turns still
underflow a 200 ms client buffer by more than 100 ms at 20 callers, and p95
buffered stall is 0.491 s against the 0.1 s qualification criterion, which is
**not met at any tested level**. What changed is that the underflow is roughly
halved and affects fewer turns.

## Where the time actually went

The round began by instrumenting the acoustic stage (`CBX_ACOUSTIC_PROFILE=1`)
rather than guessing. Under 20 callers, on the pre-existing profile:

```
95-97% of wall busy | flow 82% | vocoder 14% | watermark 3% | 1.3 rows/batch
```

Two things that measurement settled immediately:

* **The acoustic stage was saturated** — it, not stage 0, was the ceiling.
  Stage-0 inter-token latency measured 12 ms under the same load, i.e. about
  3x realtime per stream, and was never the constraint.
* **Batching was not happening.** `acoustic_max_batch_rows` was 32 and the
  decode ladder exists to make concurrent streams length-identical so they
  batch, yet the stage was averaging 1.3 rows per forward pass.

## Changes

### 1. The Perth watermark ran on the CPU, inside the stage-1 forward

Perth is a torch model — an STFT pair around a 2.4 M-parameter conv encoder —
that upstream instantiates on the **CPU**. It is applied to **every emitted
chunk of every stream**, one row at a time, inside the acoustic forward pass.
Measured cost: **40 ms for a 0.2 s chunk, 107 ms for a 6.4 s chunk**, during
which the GPU is idle.

Everything after the resample is plain torch, so it now runs on the acoustic
device. The resampler that defines the signal band (librosa/soxr) stays on the
CPU, so that part is bit-identical.

| audio | CPU | GPU | speedup | CPU-vs-GPU max diff | watermark's own perturbation |
|---|---|---|---|---|---|
| 0.2 s | 6.4 ms | 1.8 ms | 3.5x | 6.8e-06 | 0.066 |
| 1.6 s | 28.6 ms | 2.6 ms | 11.0x | 5.4e-06 | 0.074 |
| 6.4 s | 107.2 ms | 5.4 ms | **19.8x** | 6.1e-06 | 0.080 |

The placement changes the waveform by at most **7e-6 — about 1e-4 of the
watermark's own perturbation of the signal**, and ~57x smaller than the 3.97e-4
deviation this port already ships from estimator compilation. Under the gate's
`inference_mode` the measured ratio is tighter still:

```
[perth 0.2s] cpu-vs-gpu=2.384e-07 watermark=6.444e-02 ratio=3.70e-06
[perth 6.4s] cpu-vs-gpu=2.980e-07 watermark=8.505e-02 ratio=3.50e-06
```

In the profile the watermark fell from roughly half of stage-1 wall time to
**3%** (4 ms/batch).

A CPU fallback (`acoustic_watermark_device: cpu`) applies the batch's rows
across a thread pool instead; that path is **bit-identical** to the serial one
(verified over 1/4/8/16/32 rows) and is 3-11x faster than serial.

Also fixed: an empty emitted chunk (the crossfade holdback can consume a short
one entirely) crashed Perth's STFT with a reshape error, which would have taken
the acoustic engine down. Empty chunks now pass through unmarked.

### 2. The vocoder ran per row while the flow solver was batched

`decode_batch` batched the flow solver and then looped over rows for the
vocoder. Every row in a group shares its code length *and* its prompt length
(both are in `batch_key`), so the mel slices have identical geometry and the
vocoder can run once for the batch. The per-row RNG is preserved exactly:
`_draw` already accepts one generator per row and draws each row's noise at that
row's own shape, taking the same values in the same order as the solo call.

Measured against the shipped path, at 2-8 rows: **1.15x** on its own, **1.50x**
combined with the GPU watermark, with max deviation **7.1e-5 relative to peak**
— still below the 3.97e-4 this port already ships from estimator compilation.

A new gate (`test_acoustic_serving_paths.py`) decodes each row alone and inside
a three-row two-voice batch with the batched vocoder on, and compares:

```
[batched-vocoder A] batched-vs-solo max|d|=3.852e-04 (0.0421% of peak)
[batched-vocoder B] batched-vs-solo max|d|=5.826e-05 (0.0098% of peak)
[batched-vocoder C] batched-vs-solo max|d|=4.965e-04 (0.0587% of peak)
```

Stated plainly: this is **3-4x larger than the 0.008-0.017% the flow-only
batching produces**, because the vocoder's convolutions now also pick algorithms
per batch shape. It passes the same 0.1%-of-peak isolation bar the port applies
to mixed-voice batching, and the two voices remain distinguishable, but it is a
real widening of batch-composition dependence rather than a free win.

### 3. The chunk ladder: fewer, larger decodes — which is also *better* quality

Every chunk decode re-renders the reference prompt (250 codes) plus its window.
The cost is therefore dominated by the **number** of decodes, not their length —
confirmed by the profile, where flow cost stayed ~90-100 ms per call across very
different sequence lengths. A steeper ladder makes fewer, larger decodes, and
each one is rendered with *at least as much* context as before, so it reduces
work without reducing context.

Ladder changed from `first=5, growth=1.8, cap=100` to `first=10, growth=2.5,
cap=300`, window 256 → 400. Acoustic busy fell from 95-97% to 56-79%.

Against the port's own streaming gate (streamed vs one-shot, same CFM 4, FP32):

| | old ladder | new ladder | one-shot |
|---|---|---|---|
| `en_plain` chunks / cost | 5 / x2.41 | 3 / **x1.59** | — |
| `en_plain` CER | 0.0 | **0.0** | 0.0 |
| `en_plain` speaker similarity | 0.8308 | **0.8448** | 0.8689 |
| `en_long` chunks / cost | 7 / x2.92 | 4 / **x1.66** | — |
| `en_long` CER | 0.0 | **0.0** | 0.0 |
| `en_long` speaker similarity | 0.8965 | **0.9048** | 0.9105 |

Both ladders match the one-shot length exactly (0 samples). CER is 0.0 for both.
Speaker similarity **improves** and moves closer to the one-shot reference,
which is what the "more context per chunk" argument predicts. The seam metric
rises modestly (1.2 → 1.33 and 1.22 → 1.51, against one-shot 1.05 and 1.27):
there are fewer joins, but each spans a larger context shift. That is the one
metric where the new ladder is behind, and it is recorded rather than hidden.

### 4. The first chunk was too small to survive its own gap

Tracing chunk arrivals under load showed **all** the playback starvation sat in
the first two chunk transitions; chunks 3 and beyond never underflowed.

The cause: the encoder's 3-code lookahead is subtracted from the first block, so
a 5-code first block **emits 2 codes — 60 ms of audio**. The client starts
playing 60 ms and cannot survive until the second chunk arrives.

```
first block 5   chunk0 @0.44s = 0.06s audio -> chunk1 @0.70s : 0.46s underflow
first block 10  chunk0 @0.41s = 0.26s audio -> chunk1 @0.73s : 0.00s underflow
```

Raising the first block to 10 costs TTFA linearly (one code is one stage-0 step)
and cut stall p95 from 2.33 s to 1.21 s. Going further to 15 cost more TTFA
(p95 1.06 → 1.42 s) for almost no additional stall benefit, so 10 was kept.

### 5. Supporting fixes

* **Cold start was materially worse than warm, and the prewarm was warming the
  wrong shapes.** `warm_acoustic_graphs` captured a hardcoded rung list
  (5, 15, 35, 75, ...) belonging to the *old* ladder, and the profile passed it
  an empty prompt list so it did nothing at all. Every graph shape was therefore
  captured during the first requests that hit it. Measured at 20 callers on the
  same profile: p95 buffered stall **1.184 s cold against 0.588 s warm**, and
  TTFA p95 1.267 s against 0.757 s. The prewarm now derives its rungs from the
  ladder the deploy actually runs and is enabled with the 250-code reference
  prompt, capturing 8 graphs before admission. A cold server then measures
  **0.671 s stall / 0.840 s TTFA p95** — close to warm, and with the *lowest*
  share of turns stalled of the three (30.5%).
* **Graph cache saturation.** `acoustic_graph_cache_size` was 12 and filled
  within 11 seconds of startup; every shape after that silently ran eager.
  Batching the vocoder multiplies the shape space, so this became load-bearing.
  Raised to 96, and the eager-fallback count is now a logged counter rather than
  a silent condition. Widening `acoustic_graph_max_codes` from 75 to 300 to
  cover the new ladder's mid rungs was **tried and rejected** — capture overhead
  exceeded the saving (stall 0.885 → 1.550 s at 20 callers).
* **Ladder consistency across processes.** Stage 0 transports on the ladder and
  stage 1 decodes on it, and the connector rejects a mismatch. They are separate
  processes reading different sections of the profile, so both `serve.sh` and
  the supervisor wrapper now derive the three values from the profile's
  connector block — the profile is the single source of truth and the two launch
  paths cannot drift.
* **Scheduler batch limits** are now configurable
  (`acoustic_fresh_batch_size`, `acoustic_continuation_batch_size`). Raising
  them from 4/2 to 8/8 did **not** move rows-per-batch (1.3 → 1.4): once the
  acoustic stage is no longer saturated there is no queue to batch from. Kept at
  8/8 as headroom, but it is not what produced the result.

### 6. A latent hazard the new gate uncovered: cuDNN TF32 is disabled globally

`ChatterboxS3Gen.__init__` calls `_disable_conv_tf32()`, which switches cuDNN
TF32 off **process-wide**. That is deliberate and correct for the acoustic stage
— it is what stops a row's audio depending on who else is in its batch — but it
is not scoped to the acoustic model, and the conditioning encoder's convolutions
are sensitive to it:

```
cudnn.allow_tf32=True   speaker_emb rel=0.000e+00   embedding rel=0.000e+00   (bit-identical to reference)
cudnn.allow_tf32=False  speaker_emb rel=6.015e-04   embedding rel=2.933e-02   (293x the Gate A threshold)
```

So whichever runs first decides whether `ConditioningEncoder` reproduces the
reference speaker embedding exactly or drifts ~3%.

**This is not a live defect in the shipped deploy**: the encoder runs in the
input-processing process and the acoustic model in stage 1, and stage 1 only
reshapes the transferred conditioning payload rather than recomputing it
(`_reference_from_payload` runs no encoder). It is a hazard for any
single-process use, and it made the test suite order-dependent — adding a test
file that sorts before `test_gate_a_conditioning` and builds an acoustic model
silently failed Gate A. The new test file restores the flag on teardown so the
suite is order-independent again; the underlying global was left alone because
it is a deliberate correctness setting, and narrowing its scope is a change that
deserves its own qualification rather than a late edit.

## What was rejected, with numbers

* **FP16 flow estimator.** Changed the waveform by up to 0.267 max-abs (vs FP32)
  at 155 codes, for 22% of stage-1 GPU time at 8 rows. Rejected as a quality
  trade; the throughput was recovered elsewhere instead.
* **Three flow-matching steps.** 13% of stage-1 GPU time. Restored to four.
* **`acoustic_graph_max_codes: 300`** — see above.
* **First block 15** — TTFA p95 1.06 → 1.42 s for stall 1.21 → 1.12 s.

## Verification

* **The entire test suite passes: 114 tests, 0 failures.** The golden captures
  for both reference voices were regenerated from the official runner first
  (`capture_golden.py --voice ... --voice-id ex01|ex02`, 12 cases each), so this
  run includes the full Gate A/B/D fidelity gates against the reference
  implementation, not just the streaming subset. Three of those gates bear
  directly on the vocoder batching and now actually execute:
  `test_row_order_does_not_change_a_row`,
  `test_mixed_voices_share_a_batch_and_each_row_is_exact` and
  `test_different_voices_are_never_merged_into_one_batch` — each row in a mixed
  batch still matches its solo decode, and a row's audio does not depend on who
  else is in its batch. `test_waveform_matches_reference_for_identical_codes`
  passes for every case including `ar_plain`, `mixed_ar_en` and `en_short`.
  Run them from the runtime checkout (`cd /workspace/repos/vllm-omni`); pytest
  from the overlay root imports its incomplete namespace and fails collection.
* **A new gate covers the paths this profile actually turns on.** The existing
  Gate D isolation tests build `ChatterboxS3Gen()` with the batched vocoder off
  and the CPU watermarker, so they did not exercise the shipped configuration.
  `test_acoustic_serving_paths.py` (5 tests) re-proves row isolation and order
  independence with `batch_vocoder=True`, pins the thread-pool watermark as
  bit-identical to the serial path, pins the empty-chunk passthrough, and bounds
  the GPU watermark against the CPU one.
* **Voice isolation at 16 concurrent**: min margin 0.209, own-voice similarity
  0.878 vs other-voice 0.602, 16/16 succeeded.
* **Determinism**: same seed reproduces the same length and waveform within one
  PCM16 quantization step (3.05e-05).
* **Cancellation**: the server continues serving after a mid-stream cancel.
* **Five functional smoke renders** (EN, AR, mixed AR/EN, short, long) keep their
  exact durations; ASR transcripts match the previous configuration's on all
  five modulo one trailing apostrophe and a one-character Arabic variant that
  the previous configuration also mis-transcribed.

## Limits and honest caveats

* These are localhost measurements with one specific workload, warmed reference
  voices and three reference clips. They are not a production capacity
  guarantee, and no held-out multi-language quality corpus was run.
* **Run-to-run variance on this metric is large.** Three 3-minute runs of the
  identical 20-caller configuration produced p95 buffered stalls of 0.514,
  0.588 and 0.671 s. Differences below roughly 0.1 s between configurations in
  this report should not be read as real.
* **A strict real-time criterion is still not met**: 26.8% of turns at 20
  callers underflow a 200 ms buffer by >100 ms, and p95 buffered stall is
  0.514 s against a 0.1 s criterion. A client buffer of ~0.6-1 s would absorb
  what remains; a 200 ms one will not.
* The remaining stall is concentrated in `greet` turns — the first turn of each
  call, which arrive in bursts — at 71% stalled versus 24% for long `explain`
  turns. That is a queueing effect at call setup, not steady-state capacity.
* The 16-caller row is worse than the 20-caller row on both stall and TTFA, and
  is the only level where stall regressed against the previous profile. That is
  run-to-run variance at the ramp, not a real inversion; it is left in rather
  than re-rolled.
* Acoustic decode averages ~1.1-1.4 rows per batch. The flow solver is
  launch-bound at these sizes (~90-100 ms per call largely independent of
  length), so the largest remaining lever is getting more rows into each pass —
  which needs either a queue (saturation) or a scheduler that groups by decode
  rung rather than by transport geometry.

## Reproduce

```bash
supervisorctl restart chatterbox            # or:
DEPLOY_CONFIG=vllm_omni/deploy/chatterbox_mtl_v3_realtime.yaml LOG=- port/tools/serve.sh

LOAD_TTS_SEED=42 LOAD_PACE_PLAYBACK=1 LEVELS="16 20 24 30" DUR=3m \
  OUT=port/artifacts/latency/final/sweep port/tools/sweep_concurrency.sh

# where the stage-1 wall clock goes, live
CBX_ACOUSTIC_PROFILE=1 supervisorctl restart chatterbox
grep -o "\[cbx-acoustic\].*" /var/log/portal/chatterbox.log

# per-chunk arrival trace for one request (finds playback underflow directly)
python port/tools/trace_chunks.py label
```

Note that `CBX_ACOUSTIC_PROFILE=1` synchronises the device on every phase
boundary to attribute GPU time; it is a diagnostic and it slows the server down.
Every number in the table above was measured with it **off**.
