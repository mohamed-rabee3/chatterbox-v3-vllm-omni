# Chatterbox Multilingual V3 on vLLM-Omni

A production-oriented port of **Chatterbox Multilingual V3** (T3 autoregressive
codec LM → S3Gen flow-matching + HiFT vocoder) to **vLLM-Omni**'s two-stage
pipeline, with concurrent serving and **real incremental audio streaming**.

Every number in this repository was measured on the hardware described below.
Where something was tried and did not work, the measurement is kept rather than
deleted — [`progress.md`](progress.md) is the engineering log, including the
negative results.

## Current latency work

`chatterbox_mtl_v3_realtime.yaml` is the current profile for live voice agents
at ~20 concurrent callers. It holds quality at the qualified settings — FP32
throughout, four flow-matching steps, the reference watermark on every emitted
chunk — and takes its speed from removing serialization: the Perth watermark's
DSP moved to the acoustic device, the vocoder batched alongside the flow solver,
and a steeper chunk ladder that makes fewer, larger decodes (which measurably
*improves* speaker similarity rather than costing it). See
[the report](port/artifacts/latency/final/README.md) for the measurements, the
rejected experiments and the remaining limits — including that a strict
real-time criterion at a 200 ms client buffer is still not met.

The earlier `chatterbox_mtl_v3_low_latency.yaml` round is documented in
[the previous latency report](port/artifacts/latency/README.md). Install the
full overlay, including its acoustic scheduler, when using either profile.

## What this is

The original qualification measured these two profiles (historical recordings
and runtime; see the new latency report below for the current instance):

| | `chatterbox_mtl_v3.yaml` | `chatterbox_mtl_v3_streaming.yaml` |
|---|---|---|
| policy | decode the completed clause | incremental, 5-code first rung, ×2 ladder |
| flow-matching steps | 10 | 4 |
| TTFA, 4 s reply, idle | ~3.0 s | **~0.30 s** |
| TTFA, 10 s utterance, idle | ~9.8 s | **~0.30 s** |
| TTFA p95 @ 30 calls | 46.8 s | **2.15 s** |
| throughput | 1.53 turns/s | ~1.10 turns/s |

Pick by workload: the default for batch/offline throughput, streaming for live
agents. Neither is a flag flip of the other.

## Original fidelity qualification

These results precede the current optimization run. The latency report records
waveform comparisons and validation for the new profile.

* T3 prefill embeddings **bit-identical** to the reference implementation (0.000e+00).
* Acoustic waveform within the reference's own run-to-run variation (2e-7…5e-7).
* ASR **CER 0.0000** on EN and AR against the reference.
* Streaming CER matches the one-shot decode; speaker similarity within 0.019.

## Layout

```
vllm_omni/          drop-in overlay: new files at their upstream paths
  model_executor/models/chatterbox_mtl_v3/    model, sampling, streaming, s3gen, vendor
  model_executor/stage_input_processors/      T3 → S3Gen stage transport
  entrypoints/openai/tts_adapters/            OpenAI /v1/audio/speech adapter
  transformers_utils/configs/                 model config
  core/sched/                                 acoustic request scheduling
  deploy/                                     deploy profiles
patches/            the 11-line registry edits against upstream vllm-omni
port/tests/         offline gates and streaming regression tests (pytest)
port/tools/         export, serve, benchmark, load-test and diagnostic scripts
port/reference/     reference runner + golden capture
port/artifacts/     measurement results (JSON) — the evidence behind the claims
docs/               the original port plan
```

## Install

Requires a working **vLLM-Omni** checkout with vLLM 0.28.0. The low-latency
profile is tested against commit `b3dd45874a750f7edfa39bb02262804228e5ff7b`;
its scheduler integration must be reviewed when updating upstream.

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
git checkout b3dd45874a750f7edfa39bb02262804228e5ff7b
# 1. drop in the new files
rsync -a /path/to/this/repo/vllm_omni/ vllm_omni/
# 2. register the model, pipeline, adapter and config
git apply /path/to/this/repo/patches/0001-register-chatterbox-mtl-v3.patch
```

Export the checkpoint into the layout the server expects:

```bash
python port/tools/export_model.py --out /workspace/models/chatterbox-mtl-v3
```

## Run

```bash
# real-time voice agent (20 concurrent callers)
PORT=18091 DEPLOY_CONFIG=vllm_omni/deploy/chatterbox_mtl_v3_realtime.yaml \
  port/tools/serve.sh
# or the throughput profile:
PORT=18091 port/tools/serve.sh --deploy-config chatterbox_mtl_v3.yaml
```

`serve.sh` derives the streaming chunk ladder from the profile's connector
block and exports it, because stage 0 transports on that ladder and stage 1
decodes on it and the two run as separate processes.

```bash
curl -N -X POST http://127.0.0.1:18091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"/workspace/models/chatterbox-mtl-v3","input":"Hello there.",
       "language":"en","ref_audio":"data:audio/wav;base64,<REF_WAV_B64>",
       "response_format":"pcm","stream":true,"stream_format":"audio"}' --output out.pcm
```

## Verify

```bash
python -m pytest port/tests -q          # requires regenerated golden fixtures
python port/tools/smoke_client.py       # languages, rejection paths, streaming
python port/tools/test_concurrency.py   # voice isolation, determinism, cancellation
LEVELS="8 16 30" port/tools/sweep_concurrency.sh   # capacity vs TTFA
locust -f port/tools/locustfile_conversation.py --headless -u 30 -r 2 -t 5m \
       --host http://127.0.0.1:18091    # 30 concurrent conversations
```

The gates need golden captures, which are **not** committed (they are ~43 MB of
regenerable `.npz`). Regenerate them from the reference implementation:

```bash
python port/reference/capture_golden.py --voice ex01
python port/reference/capture_golden.py --voice ex02
```

Reference voice clips are likewise not committed; `port/tools/` expects them at
`/workspace/refvoices/en_{ex01,ex02,spk1272}.wav`.

## Measured capacity

One RTX 5090, F32, eager, single replica, streaming profile:

| concurrent calls | TTFA p50 | TTFA p95 | turns/s |
|---|---|---|---|
| 8 | 0.53 s | **1.00 s** | 0.84 |
| 12 | 0.78 s | 1.20 s | 0.94 |
| 16 | 0.97 s | 1.37 s | 1.07 |
| 20 | 1.21 s | 1.63 s | 1.06 |
| 30 | 1.73 s | 2.15 s | 1.10 |

p95 under 1 s at 8 concurrent calls; p50 under 1 s up to 16. Past ~30 the server
is compute-saturated and extra callers only queue. Those are historical measurements, not a hardware lower bound. The current
optimization results and remaining limits are in the latency report.

## Known limitations

Stated in full in [`progress.md`](progress.md); the short list:

* Streaming does not reproduce the non-streaming waveform bit-for-bit — it is a
  different, equally valid rendering (the token encoder is bidirectional).
  `seed` reproduces a streamed request only against other streamed requests.
* Full-model CUDA graphs freeze Python request metadata and remain unsafe. The
  new low-latency profile uses PIECEWISE capture of the Llama backbone only; CFG,
  embeddings and request identity are refreshed outside the captured region.
* Client-observed TTFA (~0.30 s idle) is higher than the engine's own
  first-chunk clock (67–120 ms); the difference is bounded but not fully
  attributed.
* No large-scale held-out language-quality corpus; no Saudi fine-tuning.

## Licence and provenance

`vllm_omni/model_executor/models/chatterbox_mtl_v3/vendor/` is vendored verbatim
from the Chatterbox model code (MIT, LICENSE retained in that directory). Files
carrying deliberate changes are marked with `# PORT:` comments. Model weights are
**not** included in this repository.
