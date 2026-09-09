# Chatterbox Multilingual V3 on vLLM-Omni

A native two-stage port of Resemble AI's Chatterbox Multilingual V3 text-to-speech
model, with continuously batched autoregressive generation, strict classifier-free
guidance, and per-request isolation.

```
stage 0  chatterbox_mtl_v3_t3      LLM_AR          text + reference voice -> speech codec ids
stage 1  chatterbox_mtl_v3_s3gen   LLM_GENERATION  codec ids -> 24 kHz waveform
```

## Pinned revisions

| Component | Revision |
|---|---|
| Official `resemble-ai/chatterbox` | `5de7a54aa4e5e2baadb0182dde554908b48b85c2` |
| HF `ResembleAI/chatterbox` | `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` |
| vLLM core | `0.28.0` |

Every artifact's SHA-256 is in `constants.ARTIFACT_SHA256` and is verified by the
exporter before a model directory is written.

## Checkpoint profiles

The pinned official loader selects the **V3 T3** but still loads `s3gen.pt`, even
though the repository also contains `s3gen_v3.pt`. File names alone do not
establish the intended pairing, so the two are separate, independently qualified
profiles and there is deliberately **no fallback between them**:

| profile | T3 | S3Gen |
|---|---|---|
| `official_loader_v3` (default) | `t3_mtl23ls_v3.safetensors` | `s3gen.pt` |
| `candidate_s3gen_v3` | same | `s3gen_v3.pt` |

The acoustic checkpoint is recorded in every exported manifest.

## Running

```bash
python /workspace/port/tools/export_model.py --out /workspace/models/chatterbox-mtl-v3
/workspace/port/tools/serve.sh
```

`serve.sh` sets three environment variables that this GPU/stack requires; see the
comments in that file for why each one is needed.

Request:

```json
{
  "model": "<served model id>",
  "input": "حياك الله، موعدك بكرة الساعة التاسعة صباحًا.",
  "language": "ar",
  "ref_audio": "data:audio/wav;base64,...",
  "response_format": "wav",
  "seed": 1234,
  "extra_params": {"cfg_weight": 0.5, "exaggeration": 0.5}
}
```

`language` is required and takes one language id per call — the model has no
"mixed" mode. `ar` and `en` are the qualified languages; the others the model
supports are refused unless `allow_unqualified_languages` is set, and a language
whose tokenizer path needs an optional third-party normalizer is refused when
that package is missing (upstream only logs a warning and silently changes
pronunciation).

`extra_params` accepts `cfg_weight`, `exaggeration`, `temperature`, `min_p`,
`top_p`, `repetition_penalty` and `policy`. Anything else is **rejected**, as are
`speed` and `instructions` — silently ignoring a parameter returns audio that is
not what was asked for.

## The contracts that are load-bearing

**Prefill layout.** `[COND placeholder]×34 | text ids | BOS | BOS`. The two BOS
embeddings are reference behaviour (`prepare_input_embeds` adds one and the
explicit decode loop appends another) and both sit at learned speech position 0.
Removing the duplicate changes the model's input.

**Learned positions come from the request's own progress.** `embed_input_ids`
fills only the conditioning block; `forward()` applies text/BOS/decode content
and both learned position tables from each row's global position inside its own
request. A batch-wide counter or the raw transformer position is wrong as soon as
two requests have different prompt lengths, and would break under slot
compaction, preemption and recomputation.

**Guidance is mandatory and strict.** Each request decodes as a conditional row
plus an unconditional companion with the *same* prompt (same ids, same
conditioning, same length); only the role differs, and the model zeroes the text
**content** for the uncond row while keeping the text **positions**. The blend is
`g = l_u + (1+w)(l_c - l_u)`, identical to Chatterbox's `g = l_c + w(l_c - l_u)`.
A row whose companion is lost is **never** released to continue unguided: it is
terminated and the request fails.

**Sampling order** matches the reference exactly, because vLLM 0.28 runs a
non-argmax-invariant processor in precisely the right place:

```
allowed-token mask -> ChatterboxCFGLogitsProcessor -> builtin penalties (pinned to 1.0)
    -> temperature -> min-p -> top-p/top-k -> draw
```

The processor does the CFG blend and the **speech-only** repetition penalty over
`[speech BOS] + generated ids`. The engine's builtin penalty would also cover the
prompt, so it is required to be exactly 1.0.

**Registering the processor is not optional.** It must be listed under stage 0's
`logits_processors` in the deploy profile. Omitting it does not fail — it silently
produces unguided, unpenalised generation.

**Reference voices are immutable and content-addressed.** Conditioning is keyed by
the decoded audio's content plus the model revision, checkpoint profile and
preprocessing version — never by path, which survives the file being replaced. A
request holds a lease so eviction cannot pull a voice out mid-synthesis.

**Three different reference windows**, all preserved: S3Gen conditioning uses the
first 10 s at 24 kHz, the T3 speech prompt the first 6 s at 16 kHz capped at 150
codes, and the T3 speaker embedding the **full** 16 kHz signal. The reference also
uses two different resamplers (librosa for the T3 window, torchaudio inside
`embed_ref`), and both are kept.

**Acoustic batching groups by length, not by voice.** Different reference voices batch
together safely — each row carries its own prompt tokens, prompt mel, speaker
embedding and noise, and a gate asserts every row matches its solo decode
(measured drift 0.008–0.017% of peak). Different *lengths* may not: padding a
short row alongside a longer one changes the short row's waveform by **40% of
peak**, because this checkpoint's token encoder does not isolate padded rows
(its output drifts 13% even when the row's own attention mask is correct).
`ragged_batching=True` exists for experimentation and a test pins that
corruption so it cannot be re-enabled by accident. The practical consequence is
that the acoustic stage runs near batch-1 in production and caps around 10×
real time per replica; raising that needs more stage-1 replicas, a compiled
estimator, or a qualified change to the encoder's masking.

**cuDNN TF32 is disabled for the acoustic stage.** With it on, decoding the same
request alone versus batched with others changed its waveform by 2.4% of peak,
because TF32 convolutions pick different algorithms per batch shape and the error
is amplified through 10 ODE solver steps and the vocoder. This is a correctness
setting, not a tuning knob.

## What is deliberately off

* **Incremental acoustic decoding** (`enable_incremental_acoustics`, default
  `False`). The acoustic decoder's output for a prefix can change when more codes
  arrive, so it needs its own prefix-stability and listening evaluation before it
  can be enabled. Until then the server streams **completed clauses**.
* **Prefix caching** — stage-0 prompt ids are conditioning placeholders plus text
  ids, so a key built from ids alone can collide across voices and exaggerations.
* **CUDA graphs on stage 0**, **speculative decoding**, **quantization**, **BF16**
  — each needs its own gate; the fidelity gates were measured in F32.
* **Chunked prefill** — it would split one pair member's prompt across steps while
  the other fits in one.

## Vendored code

`vendor/` is a verbatim copy of the official `src/chatterbox/models/` tree (MIT,
license retained). Upstream pins torch 2.6 / transformers 5.2, which must not be
resolved into the engine environment. Every divergence carries a `# PORT:`
comment; `python /workspace/port/tools/check_vendor_drift.py` fails on any
unexplained drift.

## Tests

`/workspace/port/tests/` — Gate A (tokenizer, conditioning, prefill layout),
Gate B (sampling math against HuggingFace's own processors), Gate D (acoustic
parity against the official implementation), and the stream-integrity contract.
`/workspace/port/tools/` holds the live-server suites.
