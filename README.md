# Chatterbox Multilingual V3 on vLLM-Omni

A production-oriented port of **Chatterbox Multilingual V3** (T3 autoregressive
codec LM → S3Gen flow-matching + HiFT vocoder) to **vLLM-Omni**'s two-stage
pipeline, with concurrent serving and **real incremental audio streaming**.

Every number in this repository was measured on the hardware described below.
Where something was tried and did not work, the measurement is kept rather than
deleted — [`progress.md`](progress.md) is the engineering log, including the
negative results.

## What this is

Two separately qualified deploy profiles, each with a measured trade:

| | `chatterbox_mtl_v3.yaml` | `chatterbox_mtl_v3_streaming.yaml` |
|---|---|---|
| policy | decode the completed clause | incremental, 5-code first rung, ×4 ladder |
| flow-matching steps | 10 | 4 |
| TTFA, 4 s reply, idle | ~3.0 s | **~0.30 s** |
| TTFA, 10 s utterance, idle | ~9.8 s | **~0.30 s** |
| TTFA p95 @ 30 calls | 46.8 s | **2.15 s** |
| throughput | 1.53 turns/s | ~1.10 turns/s |

Pick by workload: the default for batch/offline throughput, streaming for live
agents. Neither is a flag flip of the other.

## Fidelity

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
  deploy/                                     deploy profiles
patches/            the 11-line registry edits against upstream vllm-omni
port/tests/         84 offline gates (pytest)
port/tools/         export, serve, benchmark, load-test and diagnostic scripts
port/reference/     reference runner + golden capture
port/artifacts/     measurement results (JSON) — the evidence behind the claims
docs/               the original port plan
```

## Install

Requires a working **vLLM-Omni** checkout (this port targets vLLM 0.28 / vLLM-Omni `dev`).

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
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
PORT=18091 port/tools/serve.sh --deploy-config chatterbox_mtl_v3_streaming.yaml
# or the throughput profile:
PORT=18091 port/tools/serve.sh --deploy-config chatterbox_mtl_v3.yaml
```

```bash
curl -N -X POST http://127.0.0.1:18091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"/workspace/models/chatterbox-mtl-v3","input":"Hello there.",
       "language":"en","ref_audio":"data:audio/wav;base64,<REF_WAV_B64>",
       "response_format":"pcm","stream":true,"stream_format":"audio"}' --output out.pcm
```

## Verify

```bash
python -m pytest port/tests -q          # 84 offline gates
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
is compute-saturated and extra callers only queue. See `progress.md` for why
sub-1 s at 20–30 is not reachable on a single GPU with quality held fixed, and
what would get there.

## Known limitations

Stated in full in [`progress.md`](progress.md); the short list:

* Streaming does not reproduce the non-streaming waveform bit-for-bit — it is a
  different, equally valid rendering (the token encoder is bidirectional).
  `seed` reproduces a streamed request only against other streamed requests.
* CUDA graphs are **incompatible** with the CFG pairing implementation and must
  stay off on stage 0.
* Client-observed TTFA (~0.30 s idle) is higher than the engine's own
  first-chunk clock (67–120 ms); the difference is bounded but not fully
  attributed.
* No large-scale held-out language-quality corpus; no Saudi fine-tuning.

## Licence and provenance

`vllm_omni/model_executor/models/chatterbox_mtl_v3/vendor/` is vendored verbatim
from the Chatterbox model code (MIT, LICENSE retained in that directory). Files
carrying deliberate changes are marked with `# PORT:` comments. Model weights are
**not** included in this repository.
