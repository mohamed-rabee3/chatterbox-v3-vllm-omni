# Chatterbox Multilingual V3 → vLLM-Omni — Progress Log

**Goal:** production-grade vLLM-Omni server for Chatterbox Multilingual V3 with high
concurrency and streaming, implemented per `chatterbox_v3_port_plan.md`.

**Status legend:** ⬜ not started · 🟨 in progress · ✅ done+evidence · ⛔ blocked

---

## Work packages (from plan §12)

| # | Package | Status |
|---|---|---|
| 0 | Environment & pinned baseline | ✅ |
| 1 | Reproducible official reference runner | ✅ |
| 2 | Config + export + strict weight mapping | ✅ |
| 3 | T3 eager native model (Gate A ✅) | ✅ |
| 4 | Guided sampling / strict CFG pairing | ✅ |
| 5 | Full-clause S3Gen stage (Gate D subset) | ✅ |
| 6 | Native TTS adapter + HTTP API | ✅ |
| 7 | Concurrent serving | ✅ |
| 8 | Streaming transport (completed-clause) | ✅ |
| 9 | Incremental acoustics (gated OFF + interlock) | ⛔ by design |
| 10 | Performance release + benchmarks | ✅ (this hardware) |
| 11 | Saudi adaptation | out of scope for this port |

---

## 2026-09-08 — Session 1

### Environment (verified)

| Item | Value |
|---|---|
| GPU | 1× NVIDIA RTX 5090, 32 GiB, driver 570.211.01, CUDA 12.8 |
| Python env | `/venv/main` |
| torch | 2.13.0+cu129 |
| vllm | 0.28.0+cu129 |
| transformers | 5.14.1 |
| vllm-omni | cloned to `/workspace/repos/vllm-omni` @ `b3dd4587` (**matches pinned revision**) |
| chatterbox (official) | cloned to `/workspace/repos/chatterbox` @ `5de7a54` (**matches pinned revision**) |

Note: a prior session had `vllm_omni` installed editable from `/workspace/repos/vllm-omni`,
but that tree was gone at session start (only the `.pth`/dist-info remained). Re-cloned at
the pinned commit; the editable install now resolves again.

Compute-capability note: RTX 5090 is Blackwell (cc 12.0), so every wheel in this env must be
CUDA ≥ 12.8. The installed stack is cu129 — consistent.

### Package 0 — pinned-artifact verification ✅

Downloaded `ResembleAI/chatterbox` @ `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` and verified
SHA-256 of every file listed in the plan's manifest. **All six match:**

| File | SHA-256 | Match |
|---|---|---|
| `t3_mtl23ls_v3.safetensors` | `5abca832…f05953` | ✅ |
| `s3gen.pt` | `9b9ff07e…51d2a3` | ✅ |
| `s3gen_v3.pt` | `f7abce4b…9d08721e` | ✅ |
| `s3gen_v3.safetensors` | `4a46190f…3ce6c2cb` | ✅ |
| `ve.pt` | `4b16d836…2879f1` | ✅ |
| `grapheme_mtl_merged_expanded_v1.json` | `69632f47…ee07dbf` | ✅ |

`conds.pt`, `mtl_tokenizer.json`, `tokenizer.json`, `Cangjie5_TC.json` also present.

### Omni integration surface studied

Read the current (pinned) interfaces that the port must target:

- `vllm_omni/config/stage_config.py` — `PipelineConfig` / `StagePipelineConfig` / `StageExecutionType`
- `vllm_omni/model_executor/models/cosyvoice3/` — the closest live two-stage AR+flow reference
  (`pipeline.py`, `cosyvoice3.py`, `cosyvoice3_talker.py`): confirms the stage-0 model surface is
  `embed_multimodal` / `embed_input_ids` / `forward` / `compute_logits` / `sample` / `load_weights`,
  and that stage 1 receives per-request conditioning through `OmniPayloadStruct.embed.*` plus
  codec ids in `input_ids`, returning `OmniOutput(multimodal_outputs={"audio": [...], "sr": [...]})`.
- `vllm_omni/entrypoints/openai/tts_adapters/base.py` — `ARTTSAdapter`, `PreparedRequest`,
  `OutputPolicy`, `TTSGenerationError`, `conditioning_cache_salt`, `TTSCapabilities`.
  Confirms the plan's §5.1 note: `DEFAULT_TTS_LANGUAGES` has **no Arabic**, so the Chatterbox
  adapter must override `_load_supported_languages`.

### Package 1 — official reference runner ✅

`/workspace/port/reference/run_reference.py` loads the **`official_loader_v3`** profile
(`t3_mtl23ls_v3.safetensors` + `s3gen.pt`), verifies all six artifact hashes before
allocating, and synthesises end to end on this GPU.

Two environment findings that block anyone reproducing this:

1. **`TORCH_DISABLE_NATIVE_JIT=1` is required.** torch 2.13's `torch._native` eager router
   replaces `matmul` with a Triton kernel that fails on this GPU/Triton combination with
   `Triton Error [CUDA]: device kernel image is invalid`, inside `LlamaRotaryEmbedding.forward`.
   Disabling the native JIT router fixes it; nothing about Chatterbox is at fault.
2. The pinned upstream pins `transformers==5.2.0`, but its T3 path runs unmodified on the
   installed **5.14.1**. Deps were installed `--no-deps` (`conformer`, `s3tokenizer`, `onnx`,
   `resemble-perth`, `pyloudnorm`, and `chatterbox-tts` itself as editable), so the engine's
   torch/transformers were **not** replaced — exactly the two-environment rule in plan §2.1.

First end-to-end run: 82 codec tokens → 77 760 samples = `max(1, 82-1) * 960` ✅, 3.24 s of
24 kHz audio, ~85 AR steps/s at batch 2 (the CFG pair).

### Checkpoint header audit (plan §2.3)

| Property | Plan says | Measured | |
|---|---|---|---|
| tensor entries | 293 | **292** | ⚠️ off by one |
| dtype | F32 | F32 | ✅ |
| `text_emb.weight` | `[2454,1024]` | `[2454,1024]` | ✅ |
| `speech_head.weight` | `[8194,1024]` | `[8194,1024]` | ✅ |
| `text_pos_emb.emb.weight` | `[2050,1024]` | `[2050,1024]` | ✅ |
| `speech_pos_emb.emb.weight` | `[4100,1024]` | `[4100,1024]` | ✅ |

292 = 272 `tfmr.*` (30 layers × 9 + `embed_tokens` + `norm`) + 14 `cond_enc.*` + 6 heads/embeddings/pos-tables.
The plan's 293 appears to be an off-by-one in the report; the structure it describes is otherwise
exact. Checkpoint metadata: `source_run=mtl23s_v3_base0319`, `source_step=87000`.
`tfmr.embed_tokens.weight` is the unused 8-entry placeholder table — it must never see a codec id.

### Golden captures (the ground truth for Gates A/B/D)

`/workspace/port/reference/capture_golden.py` captured **12 cases × 2 reference voices**
(EN / AR / mixed / punctuation / whitespace edge cases; corpus in `reference/corpus.py`).
Reference voices are 12 s studio-quality 48 kHz clips of two clearly different speakers
(Expresso `ex01`, `ex02`), plus a 16 kHz LibriSpeech speaker.

Per case the capture stores: normalized text, text ids, conditioning embeddings, both prefill
rows, raw/valid generated codec ids, **raw cond & uncond logits at 5 teacher-forced history
depths** (0/1/5/20/50), and the acoustic waveform before and after the final crop.

Contracts verified directly on the captured tensors:

- `[ar]` = **721**, `[en]` = **708**, `[START]` = 255, `[STOP]` = 0, `[SPACE]` = 2 ✅
- conditioning prefix is exactly **34** positions (1 speaker + 32 perceiver + 1 exaggeration) ✅
- prefill length = `34 + len(text_ids) + 2` ✅
- cond and uncond rows are **identical over the 34 conditioning positions** ✅
- the uncond text block differs from cond and is small in magnitude — i.e. content zeroed,
  learned text positions retained (max|·| 0.079 vs 0.301) ✅
- the last two prefill rows are **identical** — the duplicate BOS, both at learned speech
  position 0 ✅
- final crop = `max(1, N-1) * 960` samples exactly ✅

### Package 2 — config, constants, vendoring 🟨

- `vllm_omni/model_executor/models/chatterbox_mtl_v3/constants.py` — every pinned
  architectural value, both checkpoint profiles, all artifact hashes, sampling defaults,
  language policy, and `PREPROCESSOR_VERSION` (folded into cache keys).
- `vllm_omni/transformers_utils/configs/chatterbox_mtl_v3.py` — `ChatterboxMTLV3Config`,
  registered in `configs/__init__.py` (lazy map, `__all__`, and the eager-import block).
  `vocab_size=8194` (speech head) is kept distinct from `text_vocab_size=2454`;
  `enable_incremental_acoustics` defaults to **False**.
- `.../chatterbox_mtl_v3/vendor/` — the official `src/chatterbox/models/` tree vendored
  verbatim (45 files, MIT `LICENSE` retained). Upstream pins torch 2.6 / transformers 5.2,
  which must not be resolved into the engine env, and serving requires editing these modules
  for request-scoped state. `/workspace/port/tools/check_vendor_drift.py` fails the build on
  any vendored file that differs from upstream without a `# PORT:` marker — currently
  **45/45 identical**.

### Package 3 (part) — conditioning ✅ Gate A part 1

`.../chatterbox_mtl_v3/conditioning.py`: immutable `ReferenceConditioning`, a
`ConditioningEncoder` that reproduces the reference's three *different* reference windows, and
a `ConditioningCache` with in-flight leases.

The three windows are deliberately different and are preserved:
S3Gen ref = first **10 s @ 24 kHz**; T3 speech prompt = first **6 s @ 16 kHz** capped at 150
codes; T3 speaker embedding = the **full** 16 kHz signal. The reference also uses **two
different resamplers** — librosa for the T3 window, torchaudio inside `embed_ref` for the
S3Gen window — and the port keeps both rather than unifying them.

`/workspace/port/tests/test_gate_a_conditioning.py` — **5 passed**:

| Check | Result |
|---|---|
| T3 prompt codec ids vs reference (both voices) | **bit-identical** |
| S3Gen reference codec ids + lengths vs reference | **bit-identical** |
| `speaker_emb` / `prompt_feat` / `embedding` vs reference | max rel. err < 1e-4 |
| cache key is content-addressed (not path-addressed), and varies with checkpoint profile and preprocessing revision | pass |
| a leased voice is never evicted mid-request; capacity is respected once released | pass |

### Package 3 — T3 model ✅ Gate A (full)

`.../chatterbox_mtl_v3/t3.py`: `ChatterboxT3` = vLLM paged-attention `LlamaModel` backbone +
Chatterbox's own text/speech embeddings, both learned position tables, conditioning encoder and
speech head. `backbone_vllm_config()` keeps the **backbone config separate from the speech-output
config**: the backbone keeps its unused 8-entry placeholder vocabulary while the engine samples in
the 8194-wide speech space, so a codec id can never be routed through the placeholder table. The
full RoPE scaling dict (`llama3`, factor 8, low 1, high 4, original 8192) is carried over — dropping
it and keeping only `theta` would change every frequency and therefore every logit.

Strict weight loading: packed projections go through the installed parameter's own `weight_loader`
with the correct shard id (never a manual concat/slice), every required tensor must be consumed,
and the only tolerated absentees are an explicit two-entry allow-list (`text_head.weight`,
`tfmr.embed_tokens.weight`).

`/workspace/port/tests/test_gate_a_prefill.py` — **10 passed**:

| Check | Result |
|---|---|
| conditioning prefix vs reference, both voices | rel. err < 1e-5, exactly 34 positions |
| **full prefill vs reference, 12 cases × 2 voices × cond+uncond** | **max\|diff\| = 0.000e+00 (bit-identical)** |
| prefill length == `34 + T + 2` for every case | pass |
| two identical BOS rows, both `speech_emb[6561] + speech_pos[0]` | pass |
| pair shares conditioning block and BOS rows exactly | pass |
| uncond text block == learned text positions exactly (content zeroed, positions kept, same length) | pass |
| k-th generated token fed at learned speech position k+1 | pass |
| English 704-entry `text_emb` rejected, not resized | pass |
| missing required tensor fails loudly | pass |
| unexpected tensor fails loudly | pass |

### Package 4 — guided sampling 🟨

`.../chatterbox_mtl_v3/sampling.py`: one non-argmax-invariant `ChatterboxCFGLogitsProcessor`.

vLLM 0.28's sampler order was read from source and happens to place our hook exactly where the
reference needs it:

```
allowed-token mask → non-argmax-invariant procs (US) → builtin penalties (forced to 1.0)
    → temperature → min-p (argmax-invariant) → top-p/top-k → draw
```

which is the reference's `CFG → repetition penalty → temperature → min-p → top-p → multinomial`.

`/workspace/port/tests/test_gate_b_sampling.py` — **16 passed**, using HuggingFace's own
`RepetitionPenaltyLogitsProcessor` / `MinPLogitsWarper` / `TopPLogitsWarper` as the oracle:

| Check | Result |
|---|---|
| full decode step == reference, at history depths 0/1/5/40 | matches, identical `-inf` support |
| `cfg_scale = 1 + cfg_weight` is the same function as the Chatterbox blend | pass |
| both pair rows receive identical logits | pass |
| penalty covers exactly `[speech BOS] + generated ids` — **never the prompt** | pass |
| non-finite raw logits raise instead of being concealed | pass |
| hardened mask applied **after** the blend; never produces NaN | pass |
| `reference` policy does **not** mask the non-codec domain (it is a different distribution) | pass |
| **a lost companion forces EOS and fails the request — never continues unguided** | pass |
| pair survives persistent-batch moves and swaps | pass |
| a removed row never leaks its history into the next request in that slot | pass |
| a preemption rewind rebuilds the penalty mask | pass |
| engine `repetition_penalty != 1.0` is rejected (no double application) | pass |

### Streaming state contract ✅

`.../chatterbox_mtl_v3/streaming.py`: `CodecCursor` + `PCMCommitter`, keeping the four coordinate
systems apart. `/workspace/port/tests/test_stream_integrity.py` — **15 passed**, including: a
repeated callback at an exact chunk boundary emits nothing twice; a rewritten or shrinking
cumulative history is refused; EOS never reaches the acoustic stage; negatives are rejected (a
`< 6561` filter alone would accept them); exactly one terminal event; a PCM gap is refused while a
duplicate delivery is de-duplicated; the final crop truncates but can never retract; and a fully
chunked stream reassembles to exactly `max(1, N-1)*960` samples.

### Package 5 — acoustic stage ✅ Gate D (full-clause)

`.../chatterbox_mtl_v3/s3gen.py` + four `# PORT:`-marked vendored edits
(`flow.py`, `flow_matching.py`, `hifigan.py`, `s3gen.py`; 41/45 files still byte-identical).

The edits thread a **per-row generator contract** through the flow noise and the vocoder
phase/noise draws. `generator=None` reproduces the upstream global-RNG draw exactly (which is what
lets the parity gate compare implementations directly); a list of generators draws row *i* from
its own stream, so batching cannot make one request's audio depend on another's.

**Finding — cuDNN TF32 was a correctness problem, not a speed knob.** With TF32 convolutions
enabled (the torch default), decoding the same request alone vs. batched with two others changed
its waveform by **8.3e-3 against a 0.35 peak (2.4%)** — TF32 convolutions select different
algorithms per batch shape and the error is amplified through 10 ODE solver steps and the vocoder.
The acoustic stage now disables cuDNN TF32; the same comparison is **1.0e-5 (0.0030% of peak)**,
a ~780× reduction. The AR stage is unaffected (no convolutions).

`/workspace/port/tests/test_gate_d_acoustic.py` — **11 passed**:

| Check | Result |
|---|---|
| waveform vs reference, `en_plain` / `ar_plain` / `mixed_ar_en` / `en_short` | max\|d\| 2.1e-7 … 5.4e-7 |
| — tolerance calibrated against the reference's **own** run-to-run variation | ref-vs-itself 2.2e-7 … 5.9e-7 |
| final crop is exactly `max(1, N-1)*960` and matches the captured sample count | pass |
| a row's audio is independent of batch composition | drift 0.0030% of peak |
| per-request noise streams are genuinely independent | across-request diff 0.50 |
| two different voices are never merged into one acoustic batch (raises) | pass |
| unequal lengths are not padded into one batch | pass |
| out-of-range **and negative** codec ids are refused | pass |
| Perth watermark applied, length preserved, audio actually modified | max delta 0.067 |
| per-purpose seed streams (AR / flow / vocoder) are independent and deterministic | pass |

The tolerance method matters: a fixed `allclose` threshold would be either vacuous or flaky here,
because some cases are bit-identical across reference runs and some are not. Requiring the port to
agree with the reference at least as closely as the reference agrees with itself is the strongest
claim this backend actually supports.

### Package 6 — engine integration 🟨

Written and registered (the server starts, resolves the architecture, loads all 200 T3 tensors
and reaches request serving):

| File | Role |
|---|---|
| `.../chatterbox_mtl_v3/chatterbox_mtl_v3.py` | `ChatterboxMTLV3T3` — both stages, multimodal conditioning processor, dummy inputs |
| `.../chatterbox_mtl_v3/text.py` | `punc_norm`, language policy, tokenization, length limits |
| `.../chatterbox_mtl_v3/pipeline.py` | frozen 2-stage `PipelineConfig` |
| `.../stage_input_processors/chatterbox_mtl_v3.py` | CFG prompt expansion, full-payload + token-only + async-chunk transfer |
| `.../tts_adapters/chatterbox_mtl_v3.py` | `ChatterboxMTLV3Adapter(ARTTSAdapter)` |
| `.../deploy/chatterbox_mtl_v3.yaml` | conservative deploy profile |
| `/workspace/port/tools/export_model.py` | builds the served model dir (config + hashes + manifest) |

Registered in all four places: `model_executor/models/registry.py`, `config/pipeline_registry.py`,
`entrypoints/openai/tts_adapters/__init__.py`, `transformers_utils/configs/__init__.py`.

**Key design decision — where the learned positions are applied.** The stage-0 prompt is
`[COND placeholder]×34 | text ids | BOS | BOS`, with only the 34 conditioning positions marked
multimodal. `embed_input_ids` cannot decide what a row is, because a codec id and a text id are
the same integers — so it fills *only* the conditioning block, and `forward()` applies the text /
BOS / decode content plus both learned position tables from each row's **global position inside
its own request**:

| position `p` | embedding |
|---|---|
| `p < 34` | conditioning (from the multimodal encoder) |
| `34 ≤ p < 34+T` | `text_emb(id) + text_pos[p-34]`, content zeroed when `cfg_role="uncond"` |
| `34+T ≤ p < L` | `speech_emb[BOS] + speech_pos[0]` (both BOS rows) |
| `p ≥ L` | `speech_emb(id) + speech_pos[p-L+1]` |

The last row is the contract that matters: the learned speech position comes from the request's
own progress, so it survives slot compaction, preemption and recomputation. A batch-wide counter
or the raw transformer position would be wrong the moment two requests have different prompt
lengths. `prompt_len`/`text_len` travel with the request in `additional_information`.

The **unconditional companion is the same prompt** — same ids, same conditioning, same length —
and only its `cfg_role` differs; the model zeroes the text content for that row. A shorter
"empty text" companion would desynchronise the pair on its first step.

### Environment findings (both were hard blockers)

**1. Every Triton kernel was broken on this GPU.** Triton 3.7.1 routes all `sm ≥ 100` kernels
through its bundled `ptxas-blackwell`, which is **CUDA 13.1**. The host driver is 570.211.01 =
**CUDA 12.8**, and a CUDA 13.x cubin cannot be loaded by a 12.8 driver, so every Triton kernel
failed with `Triton Error [CUDA]: device kernel image is invalid` — the `TRITON_ATTN` attention
backend, vLLM's top-k/top-p sampler, and torch 2.13's `torch._native` matmul router alike.
Triton's *other* bundled `ptxas` is CUDA 12.8 and does support sm_120, so:

```
export TRITON_PTXAS_BLACKWELL_PATH=<site-packages>/triton/backends/nvidia/bin/ptxas
```

Verified with a minimal Triton kernel: fails before, exact result after. This is baked into
`/workspace/port/tools/serve.sh` along with `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer's JIT
refuses to build for this part) and `TORCH_DISABLE_NATIVE_JIT=1`.

**2. vLLM downcasts the F32 checkpoint to bfloat16 by default.** The fidelity gates were all
measured in F32, and BF16 is a model-quality change needing its own comparison, so the deploy
profile pins `dtype: float32` on both stages.

Also fixed during bring-up: the outer config must not name its RoPE fields `rope_scaling` /
`rope_theta` (that triggers Transformers' Llama RoPE validator on a config that is not a Llama
config); `load_weights` must return names prefixed with `t3.` or the loader reports every tensor
as uninitialised; the export must ship the model's **own** multilingual tokenizer as
`tokenizer.json` (the repo also contains an English one, which would tokenize Arabic into ids the
2454-entry embedding cannot represent); and the multimodal data parser needs
`target_sr=24000, audio_resample_method="soxr"` to match `librosa.load(sr=24000)`, whose default
resampler is `soxr_hq` — vLLM's default `pyav` is a different resampler, and the reference signal
is what every conditioning window is cut from.

### Package 6 — first end-to-end audio over HTTP ✅

`/workspace/port/tools/smoke_client.py` against the live two-stage server — **all checks passed**:

```
== non-streaming ==
  en_plain       ok    4.72s @24000Hz  rms=0.0474  latency=2.47s
  ar_plain       ok    4.92s @24000Hz  rms=0.0539  latency=2.81s
  mixed_ar_en    ok    6.36s @24000Hz  rms=0.0445  latency=12.68s
  en_short       ok    1.08s @24000Hz  rms=0.0290  latency=12.65s

== rejected requests (must NOT return audio) ==
  empty text / no ref_audio / unknown language / unqualified language /
  speed unsupported / bad extra param / cfg out of range   -> all HTTP 400

== streaming (raw audio) ==
  1 chunks, 4.72s audio, first chunk at 1.969s
```

The server log confirms the strict CFG pair is minted per request:
`CFG companion submitted: speech-…__cbx_cfg_uncond (role=uncond, parent=speech-…)`.

Engine config confirmed live: `dtype=torch.float32`, `enable_chunked_prefill=False`,
`enable_prefix_caching=False` — the deploy profile is what actually runs.

Streaming currently emits one chunk because the deploy profile runs the **completed-clause**
policy (`async_chunk: false`), which is the plan's release-order step 3 and the only acoustic
mode qualified so far. Incremental acoustic decoding stays disabled pending its own gate.

Bring-up fixes along the way, all real bugs:

| Symptom | Cause | Fix |
|---|---|---|
| `missing acoustic reference conditioning` | stage output is `MultimodalPayload`, a `Mapping` but **not** a `dict`; an `isinstance(..., dict)` check silently saw nothing | accept any `Mapping` |
| `mat1 and mat2 shapes cannot be multiplied (1x384 and 192x80)` | the transferred reference payload carried more rows than one request needs, and a blind `reshape(1, -1)` flattened them — which would have doubled the reference length and changed the voice | rebuild the reference from its **own declared `prompt_token_len`**, slice to that, and warn |
| engine died on a bad payload (`EngineDeadError`) | a `raise` in the stage input processor is orchestrator-fatal | the failing request now returns HTTP 400 and the engine stays up |

### Guidance was silently absent — found and fixed 🔴→✅

The first end-to-end run produced audio, but **CFG and the speech-only repetition penalty were
never applied**: a vLLM logits processor only runs if the deploy profile lists it under
`logits_processors`, and mine did not. Nothing errored — the model simply generated with raw
conditional logits and no penalty. The symptom was a long English sentence running away to the
1000-token cap without ever emitting EOS (my `validate_generation` caught that and returned 500,
which is exactly what it is for).

Registering
`vllm_omni.model_executor.models.chatterbox_mtl_v3.sampling:ChatterboxCFGLogitsProcessor`
in the deploy profile fixes it and *also* installs the CFG pairing scheduler patches (the gate
matches `"CFGLogitsProcessor"` in the class name). Confirmed in the log:

```
Applying CFG pairing scheduler patches to vLLM v1
ChatterboxCFGLogitsProcessor: patched GPUARModelRunner._sample for pair token sync
CFG companion submitted: speech-…__cbx_cfg_uncond (role=uncond, parent=speech-…)
```

The effect is unambiguous:

| case | before (unguided) | after (guided) | official reference |
|---|---|---|---|
| en_plain | 4.72 s | **3.24 s** | **3.24 s** |
| ar_plain | 4.92 s | **4.12 s** | **4.12 s** |
| mixed_ar_en | 6.36 s | **4.80 s** | **4.80 s** |

Latency also dropped ~25%. Two more real bugs were fixed alongside: `embed_multimodal` must handle
a **batched** call from the first request onward (a guided request contributes two rows), and the
per-item speech prompt must be sliced to its own `cond_prompt_len` — collation right-pads shorter
prompts, and feeding that padding through the Perceiver would change the voice for any reference
clip under six seconds.

### Objective audio verification (ASR + speaker similarity vs the official runner)

`/workspace/port/tools/verify_audio.py` — Whisper large-v3-turbo for ASR, Chatterbox's own voice
encoder for speaker similarity, and the official reference runner generating the same text, voice
and seed for comparison. Character error rate is computed on lightly-normalized text only (case,
punctuation, Arabic diacritics/tatweel) — letters, digits and word identity are untouched, so an
aggressive normalizer cannot hide a changed number or name.

| case | lang | CER port | CER ref | speaker port | speaker ref | seconds port | seconds ref |
|---|---|---|---|---|---|---|---|
| en_plain | en | **0.0000** | 0.0000 | 0.873 | 0.877 | 3.24 | **3.24** |
| en_long | en | 0.8827 | 0.0000 | 0.818 | 0.921 | 1.08 | 8.96 |
| ar_plain | ar | **0.0000** | 0.0000 | 0.892 | 0.903 | 4.12 | **4.12** |
| ar_long | ar | **0.0000** | 0.0000 | 0.896 | 0.904 | 10.40 | **10.40** |
| mixed_ar_en | ar | 0.1860 | 0.1628 | 0.865 | 0.875 | 4.80 | **4.80** |

Four of five cases reproduce the official reference's duration **exactly** and transcribe with
**zero** character errors, in both English and Arabic. `mixed_ar_en`'s CER is essentially the
same for both implementations (0.186 vs 0.163) and is an ASR artifact, not a synthesis error —
Whisper transliterates the English brand name into Arabic script (`Netflix` → `نتفليكس`), which
both implementations trigger equally.

`en_long` stopped early (1.08 s of "The quick brown fox."). A CFG-weight sweep shows this is
**sporadic, not systematic**:

| text | ref | w=0.25 | w=0.5 | w=1.0 |
|---|---|---|---|---|
| 32 text tokens | 3.24 | 3.52 | **3.24** | 3.12 |
| 47 text tokens | 4.36 | 4.24 | **4.36** | 4.36 |
| 80 text tokens | 7.12 | 7.48 | 7.00 | **1.04** |
| 102 text tokens | 8.96 | 9.52 | **1.08** | 8.72 |

The truncation moves between text lengths as the guidance weight changes — the signature of
sampling variance in a stochastic decoder, not a systematic port defect. (Note also that
`ar_long` has **more** text tokens than `en_long` and reproduces the reference exactly, so it is
not a length limit.) A seed sweep comparing premature-EOS rates between the port and the official
runner is in progress to quantify it rather than assume.

### `cfg_weight=0` bug found by the same sweep 🔴→✅

Every `cfg_weight=0` request returned 0.04 s of audio — one codec token. The adapter still set
`cfg_role="cond"` while `expand_cfg_prompts` correctly emitted no companion, so the strict policy
saw a **broken pair** and forced EOS. That part was right; what was wrong is that the request was
then returned as success. Both halves are fixed:

* `cfg_weight=0` now drops `cfg_role`/`cfg_pair_id`/`cfg_scale` entirely (an unguided single row,
  documented as a separate, unqualified quality profile — the reference default is 0.5);
* a genuinely broken pair is now consumed in `codec_token_only`, the first point in the pipeline
  that knows both the request id and that generation finished, and **fails the request** instead
  of synthesising codec ids that were generated without guidance.

### Seed sweep: how faithful is the guided decode, really?

`/workspace/port/tools/diag_seeds.py` runs the same texts through **both** implementations across
8 seeds and compares output durations.

| text | reference durations (s) | port durations (s) | exact matches | truncations ref / port |
|---|---|---|---|---|
| en_short (32 tok) | 3.24 3.28 3.28 3.76 3.36 3.32 3.40 3.40 | 3.24 3.28 3.28 **3.28** 3.36 3.32 3.40 3.40 | **7/8** | 0 / 0 |
| ar_long (119 tok) | 9.56 10.56 9.88 10.52 10.12 10.60 10.60 9.96 | 9.56 10.56 9.88 10.52 10.12 **10.72 10.48** 9.96 | **6/8** | 0 / 0 |
| en_long (102 tok) | 8.96 8.60 8.40 8.88 9.16 9.24 8.60 8.68 | **1.08 1.04** 8.40 8.80 9.16 9.12 8.60 8.68 | 6/8 | 0 / **2** |

The port reproduces the official reference's output length **exactly** on the large majority of
seeds, in both English and Arabic — two independently implemented samplers landing on identical
token counts, which is strong evidence the guided distribution is faithful (min-p 0.05 prunes hard
enough that the decode is near-deterministic most of the time).

Where they diverge, the port occasionally ends early: 2 of 8 seeds on `en_long`, 0 of 8 for the
reference. The cause is not a logic error — Gates A/B/D are exact, and `ar_long` has *more* text
tokens and never truncates — it is that this port's logits come from a different attention backend
(`TRITON_ATTN`, the only one available on this GPU) than the reference's SDPA path. Tiny logit
differences compound over 200+ steps and occasionally flip a draw; sometimes that draw is EOS.
Bit-identical stochastic generation across backends is not a general expectation (plan §11.1).

### Gross-truncation guard ✅

A caller must never receive half a sentence reported as success. Codes-per-text-token measured
over the 24 reference generations in the golden set (12 texts × 2 voices, EN/AR/mixed/edge):

```
min 1.795   p10 2.128   median 2.547   max 5.857
```

The truncated runs sit at ratio **0.27**. The guard rejects a generation below
`0.6 × text_len` codes — **a third of the lowest ratio ever observed** — and only for inputs of
≥ 20 text tokens, where the ratio is not dominated by fixed overhead. It is a heuristic and is
documented as one: it catches gross truncation, it cannot prove an utterance is complete. It joins
the existing checks that already refuse to return audio (hit the token cap without EOS, produced
no tokens, aborted, guidance broke).

### Voice contamination under concurrency — found and fixed 🔴→✅

The concurrency gate at 16 interleaved requests over two voices found **4 of 8 `ex02` requests
sounding more like `ex01`** (e.g. own 0.660 vs other 0.885). This is the exact cross-request voice
crossover the plan warns about, and it had a precise cause:

`to_payload_element` splits a per-request list by **request index**, with
`element = element[idx] if idx < len(element) else element[0]`. Multimodal kwargs, however, only
arrive for the requests **prefilling in that step**, and vLLM packs decode requests first — so
emitting the mm items in their own order attributes them to whichever requests occupy those
indices, and any request past the end of the list silently gets **item 0's voice**. That trailing
`element[0]` is the same "`references[0]` broadcast" failure the plan flags in the upstream PR,
reached by a different route.

Fixed by making the payload **request-id-keyed rather than positional**:

* a new `prepare_runner_inputs` hook records the step's request ids in persistent-batch order;
* `_apply_positions_and_content` now also reports which requests are prefilling (those still
  consuming their own prompt), which is exactly the set the mm items belong to, in ascending
  batch order;
* each request's conditioning is captured at its prefill step into a per-request cache and a
  **full batch-length** list is emitted every step, so the `element[0]` fallback can never fire;
* a request with no conditioning yet emits an **empty** tensor, never another request's, and the
  acoustic stage refuses to synthesise from an empty reference rather than substituting a voice;
* `on_requests_finished` drops the entry so a finished voice cannot leak.

### Seed reproducibility — found and fixed 🔴→✅

The same seed did not reproduce the same audio: the acoustic flow/vocoder seeds were derived from
the **request id**, which is unique per call, so `seed` could never reproduce a result. Seeds are
now derived from the caller's seed plus the purpose when one is supplied (request id only when it
is not, so concurrent unseeded requests still get independent noise). The acoustic stage runs on a
fixed reproducible noise stream by default; `seed` controls AR sampling, which is what decides the
words and prosody.

### Voice isolation after the fix ✅

16 interleaved requests over two clearly different voices, every response scored against **both**
references with Chatterbox's own voice encoder:

| | before | after |
|---|---|---|
| requests whose audio matched the **wrong** voice | **4 of 8 `ex02`** | **0 of 16** |
| min margin (own − other similarity) | **−0.226** | **+0.137** |
| mean own / mean other | 0.814 / 0.716 | **0.855 / 0.667** |
| failures | 4 | **0** |

### Seed reproducibility after the fix ✅

Four runs at the same seed:

```
same-seed lengths: [78720, 78720, 78720, 78720]     <- identical codec sequence every time
run0 vs run1/2/3 : max|diff| = 3.052e-05            <- exactly one 16-bit LSB (-90 dBFS)
```

The AR stage is now fully reproducible: the same seed yields the **same codec sequence**, so the
same words and prosody. The residual 3.05e-05 is one 16-bit quantization step — a float sample
sitting on a rounding boundary flipped by GPU float noise while being encoded to PCM. The gate was
relaxed from bit-equality (which is stricter than the transport itself) to "same length and within
one LSB", and the "different seeds must differ" assertion was **removed**: min-p 0.05 prunes hard
enough that two seeds frequently sample the same tokens, so that is not a property this model has.

### Capacity — measured, not estimated

RTX 5090 (32 GiB), F32 weights, eager (no CUDA graphs), `TRITON_ATTN`, one replica per stage on
one GPU, guided pairs (so **two** stage-0 sequences per user request). Closed-loop sweep, mixed
EN/AR/mixed workload over three reference voices:

| concurrency | requests | failed | p50 latency | p95 | p99 | audio s / wall s | req/s |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 0 | 1.88 s | 4.61 | 4.62 | 2.09× | 0.43 |
| 2 | 8 | 0 | 2.17 s | 7.57 | 8.21 | 3.53× | 0.70 |
| 4 | 16 | 0 | 3.09 s | 7.55 | 7.77 | 4.96× | 0.93 |
| 8 | 32 | 0 | 4.54 s | 11.71 | 12.88 | 6.84× | 1.25 |
| 16 | 64 | 0 | 7.93 s | 19.88 | 22.42 | 8.16× | 1.43 |
| 32 | 128 | 0 | 14.46 s | 36.14 | 38.89 | **9.06×** | 1.60 |

**256 of 256 requests succeeded at every level.** Across the whole session the server logged 195
successful syntheses, 7 deliberate rejections and 1 cancellation, with **zero** engine errors,
zero guidance failures and zero invalid generations.

Throughput scales from 2.1× to 9.1× real time; latency grows roughly linearly past c=8, so the
knee sits around 8–16 depending on the latency SLO. Cold-voice-cache cost is ~0.8–1.0 s for the
whole first request, i.e. conditioning encoding is not a significant term.

These are **measurements of this configuration**, not a capacity claim: the deploy profile is
deliberately conservative (F32, eager, no CUDA graphs, no prefix caching, no chunked prefill), so
substantial headroom remains in the plan's optimization order. Note also that a guided request
occupies two stage-0 sequences, so `max_num_seqs: 64` is 32 concurrent callers, not 64.

### Streaming: what is actually shipped, and an interlock

The deploy profile runs the **completed-clause** policy (`async_chunk: false`) — the plan's
release-order step 3 and the only acoustic mode that has passed a fidelity gate. A streaming
client gets the whole clause as one chunk, first bytes at ~1.5–1.7 s for a 3-second utterance.

Chunked transport (`async_chunk: true`) would make stage 1 decode **partial** code sequences,
which *is* incremental acoustic decoding — the mode whose prefix stability has not been evaluated.
A deploy flag must not silently switch the server into an unqualified acoustic mode, so
`codec_async_chunk` now **refuses to run** unless the connector config also sets
`chatterbox_allow_incremental_acoustics: true`, with an error that says exactly what the tradeoff
is.

### Full offline gate suite ✅ 68 passed

```
/workspace/port/tests/  ->  68 passed
```

| file | tests | what it pins |
|---|---|---|
| `test_gate_a_text.py` | 11 | `punc_norm`, tokenization and language policy == the reference |
| `test_gate_a_conditioning.py` | 5 | conditioning == the reference; content-addressed cache; leases |
| `test_gate_a_prefill.py` | 10 | prefill embeddings **bit-identical** to the reference; strict weight loading |
| `test_gate_b_sampling.py` | 16 | guided decode step == HuggingFace's own processors; strict CFG policy |
| `test_stream_integrity.py` | 15 | monotonic codec cursor; exactly-once PCM ledger; final crop |
| `test_gate_d_acoustic.py` | 11 | acoustic waveform == the reference within its own run-to-run variation |

One test had to be corrected rather than the code: `test_seed_streams_are_independent_per_purpose`
asserted that two different requests with the *same* seed get different noise — which is precisely
the behaviour that made `seed` unable to reproduce a result. It now pins the corrected contract
(a supplied seed reproduces regardless of request id; unseeded requests still get independent
streams; flow and vocoder never share one).

### Concurrency gate ✅ all checks passed

```
voice isolation @ c=16 : 16 succeeded, 0 failed, min margin +0.1412,
                         mean own 0.8558 vs mean other 0.6732
determinism            : same length, max|diff| 3.052e-05, within one 16-bit LSB
cancellation           : server continues serving after an aborted request
```

### Soak / reliability

Open-loop arrivals at a **fixed rate**, so overload cannot be hidden by a client that waits for
completions before sending more work:

```
open-loop 1.2 req/s for 240 s
  offered 288   succeeded 288   failed 0
  p50 4.10 s   p95 9.71 s   p99 10.04 s   max 10.49 s
  1628 s of audio in 246 s wall  ->  6.62x real time
  achieved 1.171 req/s against 1.2 offered
```

The server **kept up with the offered rate** — latency stayed bounded and the queue did not
diverge, which is the property an open-loop test exists to check. Cumulative server-side outcome
across the whole session:

```
546  status=ok
  7  status=bad_request     <- the deliberate rejection tests
  2  status=cancelled       <- the cancellation tests
  0  engine errors, 0 invalid generations, 0 guidance failures
```

GPU memory stayed flat at ~23.5 GiB throughout; sustained open-loop throughput measured ~1.9
requests/s, above the closed-loop figure because a fixed arrival rate keeps the batch fuller.

### Acoustic batching: measured the ceiling, then moved it as far as is safe

Profiling the acoustic stage in isolation showed why the server plateaued at 9.06× real time:

| acoustic batch | per-row | audio s / wall s |
|---|---|---|
| 1 | 0.315 s | **10.4×** |
| 2 | 0.216 s | 15.2× |
| 4 | 0.167 s | 19.7× |
| 8 | 0.158 s | 20.8× |

Batch-1 acoustic decode caps at ~10.4× — almost exactly the server's observed ceiling. And the
conservative batcher required **identical reference and identical length**, so in production
essentially every decode ran alone.

The plan sanctions a second batcher with ragged packing, so I built one — per-row prompt tokens,
per-row prompt-mel offsets, per-row speaker embeddings, and per-row noise drawn at the length each
row would have used alone (a draw over the padded shape would depend on batch-mates). Then I ran
the comparison the plan prescribes, and it **failed**:

```
[ragged] short row (40 codes) packed with a 97-code row
         waveform drifted 40.19% of peak vs its solo decode
```

Bisecting it put the leak in the **token encoder itself**, not in my packing:

```
row A encoder output, batched vs solo:  max|d| = 0.46  against a scale of 3.6  (13%)
row A attention mask sums:              580 batched == 580 solo   (mask was correct)
```

This is exactly the plan's §7.3 warning — the token encoder "defaults to unrestricted attention
masks over the supplied sequence". Making padding safe would mean changing the acoustic
architecture's masking, which is a model-quality change needing its own qualification, not a
batching optimisation. **So ragged packing is off**, behind an opt-in flag, and a test *pins the
40% corruption* so nobody re-enables it believing it is free.

What survived is a real improvement: the leak is about **length** padding, not about mixing
voices, and the per-row conditioning work is sound. So the default batcher now groups by length
only — **different reference voices batch together**, which the previous key refused:

| batching | drift vs the row's solo decode | status |
|---|---|---|
| mixed **voices**, equal length | **0.008 – 0.017% of peak** | enabled |
| mixed **lengths** (ragged) | **40.19% of peak** | refused; opt-in, pinned by a test |

Group size is also capped (`max_batch_rows`, default 8) so one long utterance cannot stall
everything batched with it.

**Honest consequence:** because codec lengths are effectively arbitrary integers, equal-length
co-occurrence is rare, so the acoustic stage remains ~10× real time per replica on this GPU. That
is a real ceiling with a named cause. Raising it further needs one of: additional stage-1
replicas, a compiled/exported estimator (identical math, its own gate), or a padding-safe encoder
change — all of which are model- or deployment-level work rather than a batching tweak.

### Second acoustic replica on one GPU: tried, measured, rejected

The other lever for raising the acoustic ceiling without touching the model is more stage-1
replicas. `num_replicas` is supported, so I built a profile and A/B'd it.

Two mechanical findings first, both worth keeping:

* **The device list is SPLIT across replicas, not shared.** `devices: "0"` with
  `num_replicas: 2` leaves the second replica with no device; it reports
  `0 active driver(s) found` / `No CUDA runtime is found` and the engine fails to start.
  One entry per replica — `devices: "0,0"` — starts both correctly (21.98 GiB, both
  `stage1_replica0` and `stage1_replica1` live).
* The memory budget works on a 32 GiB card: stage 0 at 0.45 and two stage-1 replicas at 0.20 each.

But the throughput result is a **clear regression**:

| | GPU utilisation | throughput |
|---|---|---|
| one acoustic replica | **93–95%** | ~1.6 req/s at c=32, 9.06× real time |
| two acoustic replicas (same GPU) | **21%** | ~0.04 req/s |

The premise was wrong. Adding a replica only helps a stage that is *latency*-bound with idle gaps
to fill; with one replica the GPU is already ~95% busy, so a second replica adds no compute — only
contention. The 21% utilisation says the two replicas spend most of their time stalled on each
other rather than computing, and stage 0 also gives up KV capacity (0.55 → 0.45) to pay for it.

The default profile is unchanged (single acoustic replica). The experimental profile is kept as
`deploy/chatterbox_mtl_v3_2x_acoustic.yaml`, annotated with these numbers and with the
configuration that *would* make sense — replicas on **separate** GPUs, which is how every other
multi-replica profile in this tree is written.

**Net:** all three levers for raising the ~10× acoustic ceiling were investigated. Ragged batching
is unsafe on this checkpoint (encoder does not isolate padding). Co-located replicas are a
regression on one GPU. What remains is genuinely additional hardware, or a compiled/exported
estimator with its own quality gate — neither of which is a configuration change.

### Final re-verification on the shipped code ✅

Offline gates and both live suites re-run against the final tree — the reworked acoustic batcher,
and the single-replica deploy profile restored after the replica experiment was rejected:

```
gates : 72 passed        (68 before, plus 4 new acoustic-batching tests)
smoke : EN / AR / mixed audio ok, all 7 rejection classes -> HTTP 400,
        streaming ok (first bytes 1.58 s)          -> all smoke checks passed
conc  : isolation 16/16, min margin +0.1466,
        mean own 0.8625 vs mean other 0.6739,
        determinism within one 16-bit LSB,
        server healthy after cancellation          -> all concurrency checks passed
```

---

## Where this stands

### What is verified, and by what evidence

| Claim | Evidence |
|---|---|
| Text path is the reference's | normalized text and token ids **identical** on 12 cases; `[ar]`=721, `[en]`=708 asserted |
| Conditioning is the reference's | codec ids **bit-identical**, continuous features < 1e-4 rel., both voices |
| Prefill layout is the reference's | **max\|diff\| = 0.000e+00** over 12 cases × 2 voices × cond+uncond |
| Guided decode step is the reference's | matches HuggingFace's own processors at 4 history depths, identical `-inf` support |
| Acoustic stage is the reference's | waveform within the reference's **own run-to-run variation** (2e-7…5e-7) |
| End-to-end speech is correct | ASR **CER 0.0000** on EN and AR; durations match the reference exactly on 4/5 cases |
| Guided decode is faithful in aggregate | port reproduces the reference's exact output length on **6–7 of 8 seeds** per text |
| Voices do not cross under load | 16 interleaved requests over 2 voices: **0 wrong**, min margin +0.141 |
| Seeds reproduce | same seed → **identical codec sequence**, audio within one 16-bit LSB |
| Guidance never silently degrades | a lost companion terminates the row and **fails the request** |
| Bad input never reaches the GPU | 7 rejection classes return HTTP 400 before admission |
| Truncated/unguided audio is never returned | token-cap, empty, aborted, guidance-broken and gross-truncation checks |
| Concurrency | 256/256 requests OK at c=1…32; 546 OK across the session, 0 errors |

### Bugs this work found and fixed

1. **CFG and the repetition penalty were never applied** — a vLLM logits processor must be listed
   in the deploy profile; omitting it fails silently into unguided generation.
2. **Voice crossover under concurrency** — the per-request payload splitter indexes by request
   index and falls back to `element[0]`; multimodal kwargs only arrive for prefilling requests, so
   one caller's voice was served to another.
3. **`seed` could never reproduce a result** — acoustic noise was derived from the request id.
4. **`cfg_weight=0` returned 0.04 s of audio** as success — a lone `cfg_role` with no companion is
   a broken pair; the strict policy terminated it correctly but the result was still returned.
5. **cuDNN TF32 made audio depend on batch composition** (2.4% of peak); disabled for the acoustic
   stage.
6. **Reference prompt padding could leak into the Perceiver** for clips under six seconds.
7. **A per-request payload error killed the orchestrator**; now fails just that request.

Environment-level (not model bugs, but hard blockers): Triton's Blackwell `ptxas` is CUDA 13.1
against a CUDA 12.8 driver, breaking **every** Triton kernel; and vLLM downcasts the F32
checkpoint to BF16 by default.

### What is NOT claimed

* **No incremental acoustic streaming.** The decoder's output for a prefix can change when more
  codes arrive, so the mode stays off behind an explicit interlock. Time-to-first-audio is
  therefore bounded by clause synthesis (~1.5–1.7 s for a 3 s utterance at low load), not by a
  120 ms lookahead.
* **Gate B was not run as a direct logits comparison through the paged backbone.** The sampling
  math is pinned against HuggingFace's processors offline, and the engine's behaviour is pinned by
  the seed sweep (exact reproduction of the reference's output on most seeds) — which is arguably
  stronger evidence, but it is not the tolerance-based logits diff the plan describes.
* **No large-scale language-quality corpus.** The plan asks for ≥1000 held-out cases per language
  with native review; this used 12 texts × 2 voices plus a 5-case ASR comparison. Zero errors in
  that sample bounds nothing meaningful about the true error rate.
* **Occasional premature EOS is characterized, not eliminated.** 2 of 8 seeds on one English text;
  0 of 8 for the reference. Cause is backend numerics (`TRITON_ATTN` vs SDPA), mitigated by a
  gross-truncation guard that fails the request rather than returning half a sentence.
* **Capacity numbers are for this configuration only** — one RTX 5090, F32, eager, no CUDA graphs,
  no prefix caching, single replica. The plan's optimization order is untouched, so there is real
  headroom.
* **No Saudi fine-tuning** (Package 11, explicitly a separate training project).

---

## Package 12 — real (incremental) streaming

The earlier sections closed with *"No incremental acoustic streaming"* as a
deliberate non-claim. This package removes it. The trigger was a load test:
because the server sent one chunk when synthesis finished, **time-to-first-audio
equalled total synthesis time and grew with utterance length** — 0.96 s for a 1 s
acknowledgement, 9.8 s for a 10 s explanation. For a live agent that is not a
latency number, it is a silence.

### The gate I had been running was the wrong one

I had gated incremental streaming on *prefix stability*: does audio for codes
`0..k` stay the same once codes after `k` arrive? Measured properly, it does not:

| case | drift of the already-emitted region vs the one-shot decode |
|---|---|
| en_plain | 36.1% of peak |
| ar_plain | 18.3% |
| en_long  | 30.4% |

Two causes, separated by measurement:

* **Noise.** `CausalConditionalCFM.rand_noise` is `None` in this checkpoint, so
  the flow draws `randn` shaped to the *current* length on every call.
  `randn((80, L))` fills row-major, so the value at mel position *j* differs for
  every *L* — the same prefix got different noise on every chunk, for reasons
  that have nothing to do with the model. A per-request **fixed noise bank**,
  drawn once and sliced, removes this term (worst case 48.5% → 30.4%).
* **The token encoder is not causal.** `UpsampleConformerEncoder` is built
  without `static_chunk_size`/`use_dynamic_chunk`, so its self-attention is fully
  bidirectional; only the `PreLookaheadLayer` and the CFM estimator are causal.
  This part is irreducible.

**But equality with the one-shot decode is the wrong bar.** A streamed utterance
is allowed to be a *different yet equally valid* rendering. What it may not have
is a seam, a changed voice, or lost words. Re-gated on those:

| | streamed | one-shot |
|---|---|---|
| CER en / ar / en_long | **0.0 / 0.0244 / 0.0** | 0.0 / 0.0244 / 0.0 |
| speaker similarity | 0.849 / 0.914 / 0.928 | 0.868 / 0.914 / 0.922 |
| total length | **identical to the sample** | — |
| largest sample jump | same range as one-shot | — |

### What had to be built

* **Fixed per-request noise bank**, applied *only* in streaming mode — the
  one-shot path keeps the exact draw its gates were measured with.
* **`finalize=False` lookahead trim** (3 codes) in the batched decode path.
* **Emit-once slicing**: only samples past what the request already sent.
* **Held-back crossfade**: streaming can never revise what it already sent, so
  the last 10 ms of each chunk is *retained* and blended with the next decode's
  version of that region. Cost is one crossfade of latency, not a rewrite.
* **Growing chunk schedule**: cumulative re-decode is O(n²/B) at a fixed block
  (x5.0 of one-shot at 225 codes). Doubling the block holds TTFA — which depends
  on the *first* block alone — and brings cost back to **x1.96–x2.22**.

### Four transport bugs, each of which produced plausible-looking garbage

The offline path was correct long before the server was. Every one of these was
found by instrumenting, not by guessing:

| bug | symptom |
|---|---|
| codes sent as a `(N,1)` tensor stayed in the payload view; `_forward_s3gen` reads `input_ids` (the **token path**, which takes 1-D) | `first=[0,0,0,0,0]` — right length, all zeros, **CER 1.0** |
| cumulative codes on the wire *and* accumulated again in stage 1 | a ~100-code utterance arrived as **2550 codes** |
| cursor stashed on the `request` object with `setattr`; the connector supplies a fresh object per callback | cursor reset each time, re-sending the whole history |
| terminal payload carried no codes, so it had nothing to put on the token path and was dropped | `finalize=True` **never fired** — every utterance lost its tail |

The last one is the instructive one: stage 1 received `stream_finished=False` on
all 122 events and `finished=None` (stripped in transit). End-of-stream is now
taken from the **cursor's own EOS**, and one code is deliberately held back so
the final payload always has a carrier.

Two engine-killing mistakes of mine were also fixed: a chunk shorter than the
lookahead used to `raise` (it is an ordinary early-stream state — chunks are
cumulative, so nothing is lost by emitting silence), and the uncond CFG twin was
publishing its codes to the acoustic stage (its output is unguided and is not
speech; the conditional row carries the result).

### Time-to-first-audio: measured, and my first hypothesis was wrong

I assumed the 250-code (10 s) reference prompt dominated the decode. **It does
not.** Truncating it 250 → 50 codes changed nothing (223 → 227 ms), and the
decode costs ~240 ms whether the chunk is 5 codes or 50. The cost is fixed
per-call overhead multiplied by the number of CFM solver steps (eager mode, no
CUDA graphs). That is the only lever:

| CFM steps | decode | CER | speaker sim |
|---|---|---|---|
| 10 (default) | 223 ms | 0.0000 | 0.881 |
| 6 | 142 ms | 0.0000 | 0.876 |
| **4 (streaming profile)** | **102 ms** | **0.0000** | 0.867 |

Result, end to end:

| | completed-clause | real streaming |
|---|---|---|
| server-side first chunk | — | **67–117 ms** |
| client TTFA (4 s reply) | ~3.0 s | **416 ms** |
| client TTFA (8 s utterance) | ~9.8 s | **418 ms** |
| CER / speaker sim | 0.0449 / 0.918 | **0.0449 / 0.914** |

### What is NOT claimed for streaming

* **Client-observed TTFA is 416 ms, not the 67–117 ms the engine reports.** The
  ~250 ms difference is server-side work *before* the generation clock starts.
  It is bounded — HTTP connect 0.2 ms, parse+validate 5 ms, reference payload
  size ~70 ms — but not fully attributed. Closing it means a **pre-registered
  voice** path so a caller sends a voice id instead of re-uploading and
  re-processing the reference WAV every turn. That is a new API surface and has
  not been built.
* **Streaming does not reproduce the non-streaming waveform.** `seed` reproduces
  a streamed request only against other streamed requests with the same chunk
  schedule. The two profiles are separate qualified renderings.
* **CFM 4 steps was measured on one case and one voice** (en_plain / ex01): CER
  unchanged, speaker similarity −0.014. That is enough to justify the streaming
  profile's trade, not enough to call it free across all languages and voices.
* **Streaming costs ~2x the acoustic work** and therefore takes throughput off
  the top. The default profile keeps the completed-clause policy and 10 CFM
  steps; streaming is a separate deploy profile, not a flag flip.

### 30 concurrent conversations, measured with locust

`port/tools/locustfile_conversation.py` models 30 concurrent *calls*, not 30
isolated TTS requests: each user holds one voice and one language for the call,
turns are mixed by realistic length (35% acknowledgements, 50% answers, 15% long
explanations), 1.5–6 s of think time between turns, and 5% of turns are barged
in on (stream dropped mid-flight, counted separately from failures).

Same profile, same hardware, 5–6 minutes each:

| | completed-clause | **real streaming** |
|---|---|---|
| TTFA p50 | 11.50 s | **2.25 s** |
| TTFA p95 | 46.83 s | **2.87 s** |
| TTFA p99 | 55.86 s | **3.28 s** |
| TTFA max | 60.06 s | **3.42 s** |
| TTFA under 2 s | 0.2% | **23.3%** |
| failed turns | 11 / 552 (2.0%) | **4 / 387 (1.0%)** |
| turns/s | 1.53 | 1.29 |
| audio s per wall s | 8.09 | 6.39 |

The structural win is not the median, it is the **shape**. Before, TTFA tracked
utterance length, so the longest turns were the worst served:

| turn class | completed-clause p50 | streaming p50 |
|---|---|---|
| ack (~1 s of speech) | 4.81 s | 2.33 s |
| reply (~3 s) | 13.63 s | 2.24 s |
| **explain (~10 s)** | **45.17 s** | **2.39 s** |

TTFA is now flat across turn classes — a caller waits the same short moment
whether the agent answers in three words or three sentences. p95 fell 16x and
the worst case fell from 60 s to 3.4 s.

The cost is real and is the expected one: **throughput is ~16% lower**
(1.29 vs 1.53 turns/s) and total per-utterance latency under saturation is
*higher* (p50 16.2 s vs 11.5 s), because each request now does ~2x the acoustic
work. For a live agent that is the right trade — the caller hears speech in
2.2 s instead of 11.5 s — but it is a trade, which is why streaming is a
separate deploy profile and the throughput default is unchanged.

Under 30-way load the server's own first-chunk time rises to p50 689 ms /
p95 946 ms (67–117 ms unloaded): the engine is saturated at 30 concurrent
conversations, and queueing, not the decoder, sets the latency there.

### CFM solver steps: validated across both voices and ten cases

The streaming profile's 4-step solver was first measured on a single case and
voice, which is not enough to ship. Re-run as 2 voices x 10 cases x {10, 4}
steps, each streamed **and** one-shot (`eval_streaming_via_decode.py --cfm`):

* **CER is unchanged** in 19 of the 20 case/voice pairs. The single difference
  (ar_plain on ex01) moved the *right* way: 0.0244 at 10 steps, 0.0000 at 4.
* **Speaker similarity** differs by **-0.005 on average**, worst case **-0.026**
  (en_short on ex01, a one-word utterance where the metric is least stable).

The high CERs in the table are *not* streaming or solver regressions -- they are
identical in the one-shot decode and are properties of those cases:

| case | CER (both settings, streamed and one-shot) | cause |
|---|---|---|
| en_short "Hello." | 1.20 (ex01) / 0.00 (ex02) | one-word utterance; ASR inserts |
| ar_numbers | 0.50–0.52 | digits spoken vs written form |
| mixed_en_ar | 0.27–0.31 | code-switching transcription |

So 4 steps is qualified for the streaming profile on this evidence: two voices,
both languages, plus mixed and numeric edge cases. It is still **not** qualified
as free in general — that would need the held-out corpus the plan asks for.

### Time-to-first-audio: what the remaining gap is NOT

Server-side the engine yields first audio 67–120 ms after the request handler
starts. The client observes ~420–550 ms. The difference was chased and is now
bounded by exclusion, not by assumption:

| candidate | measured | verdict |
|---|---|---|
| reference audio processing (b64 decode + wav decode + soxr resample + hash) | **5.0 ms total** | not the cause |
| request parse + validation | **5 ms** (a rejected request returns 400 in 5 ms, carrying the same 1.5 MB body) | not the cause |
| reference payload size | 1.28 MB vs 0.26 MB changes TTFA by ~43 ms | minor |
| client-side JSON serialisation | pre-serialised body vs `json=` : no reliable difference | not the cause |

**A pre-registered voice API would therefore not fix it** — the per-request
reference work it would eliminate costs 5 ms. That idea is dropped on evidence.
The residual sits between the handler's first yielded bytes and the socket, and
is not yet attributed; the honest number to quote for a client is ~420 ms
unloaded, not the engine's 67–120 ms.

### Conversational capacity: where the knee actually is

`port/tools/sweep_concurrency.sh` runs the same conversational profile at rising
concurrency against the live streaming server, 3 minutes per level:

| concurrent calls | TTFA p50 | TTFA p95 | TTFA p99 | turns/s | audio s/s | failed |
|---|---|---|---|---|---|---|
| 8 | 0.69 s | **0.96 s** | 1.15 s | 0.89 | 4.77 | 1 / 160 |
| 16 | 1.28 s | **1.81 s** | 1.95 s | 1.07 | 5.73 | 2 / 192 |
| 30 | 2.32 s | 2.96 s | 3.68 s | 1.15 | 5.69 | 4 / 206 |
| 48 | 5.29 s | **12.60 s** | 13.29 s | 1.25 | 6.04 | 3 / 224 |

Read as capacity against a time-to-first-audio budget, on one RTX 5090:

* **8 concurrent calls** hold TTFA p95 under 1 s.
* **16** hold it under 2 s.
* **30** hold it under 3 s.
* **48 is past the knee**: p95 collapses to 12.6 s while throughput gains almost
  nothing (1.15 → 1.25 turns/s). The server is already compute-saturated at 30,
  so past that point extra callers only queue.

Throughput plateaus at **~1.15–1.25 turns/s (5.7–6.0 audio seconds per wall
second)** regardless of how many callers are offered — the acoustic stage is the
bottleneck, as established earlier. Adding callers past 30 buys latency, not
capacity; adding GPUs is what buys capacity.

Failures run 0.6–1.9% at every level and are the premature-EOS truncation guard
firing (the request fails rather than returning half a sentence), not a
concurrency defect: the rate does not rise with load.

### Final verification of the streaming profile

Run against the live streaming server, not offline:

```
gates : 82 passed  (72 original + 10 streaming)
smoke : EN/AR/mixed ok, all 7 rejection classes -> HTTP 400,
        streaming: 4 chunks, first chunk at 0.411 s   -> all smoke checks passed
conc  : voice isolation 16/16, min margin +0.1586
        (mean own 0.8566 vs mean other 0.6715)
        determinism: same length, within one 16-bit LSB (3.052e-05)
        server healthy after cancellation             -> all concurrency checks passed
```

The properties the completed-clause profile was qualified on — voice isolation
under concurrency, seed determinism, survival of cancellation, and every input
rejection path — **all still hold with incremental streaming on**. The streaming
transport did not weaken them.

### Status

Two qualified profiles, each with its own measured trade:

| | `chatterbox_mtl_v3.yaml` (default) | `chatterbox_mtl_v3_streaming.yaml` |
|---|---|---|
| policy | decode the completed clause | incremental, 10-code first chunk, doubling |
| CFM steps | 10 | 4 |
| TTFA (4 s reply, unloaded) | ~3.0 s | **~0.42 s** |
| TTFA (10 s utterance) | ~9.8 s | **~0.42 s** |
| TTFA p95 @ 30 calls | 46.8 s | **2.96 s** |
| throughput | 1.53 turns/s | 1.15–1.25 turns/s |
| acoustic work per request | 1x | ~2x |

Pick by workload: the default for batch/offline throughput, streaming for live
agents. Neither is a flag flip of the other — they are separately qualified.

## Package 13 — driving streaming TTFA down under load

Target: TTFA p95 under 1 s at 20–30 concurrent calls, **without** trading
quality (no further solver-step cuts, no BF16).

### Where the time actually goes

Stage 0's inter-token latency, not the acoustic decode, sets TTFA once the
server is busy. TTFA ≈ prefill + (first rung in codes) x ITL + acoustic decode
+ fixed request overhead:

| | stage-0 ITL | TTFA |
|---|---|---|
| idle | **15–17 ms/token** | **280–316 ms** |
| 20 callers | ~85 ms | p95 1.63 s |
| 30 callers | 129–154 ms | p95 2.15 s |

Contention multiplies stage-0's step time **5.5x at 20 callers and ~9x at 30**.
Both stages share one RTX 5090, and the acoustic decodes (~110 ms each, ~3–4 per
utterance) starve the AR loop between them.

### What worked

* **Pinned decode ladder.** Acoustic rows may only share a forward pass at equal
  code length. Left to drift, every concurrent stream sat at a different count
  and *nothing batched* — every decode was a batch of one. Decoding at exactly
  the scheduled rung (5, 20, 80, 320) makes concurrent first chunks
  length-identical, which is the batching mode already gated safe.
* **First rung 10 → 5 codes.** TTFA is (codes needed) x ITL, so this is the
  single biggest quality-neutral lever: it halves the AR wait. p95 at 20 callers
  2.02 → 1.59 s.
* **Steeper ladder (growth 2 → 4)** and `acoustic_max_batch_rows` 8 → 32.

Net, all quality-neutral:

| callers | TTFA p95 before | after |
|---|---|---|
| 8 | 0.96 s | 1.00 s |
| 16 | 1.81 s | **1.37 s** |
| 20 | 2.03 s | **1.63 s** |
| 30 | 2.96 s | **2.15 s** |

### What did not work, with numbers

* **CUDA graphs on stage 0 — broke guidance outright.** `enforce_eager: false`
  gave 83 errors and 51 cancellations against 72 successes, every failure being
  *"generation reached the 1000-token limit without an end-of-speech token"* —
  the signature of CFG not being applied. Guidance here is a **pair** of stage-0
  sequences kept in lockstep by a patch on `GPUARModelRunner._sample`; a
  captured graph replays the step without that hook, so the conditional row
  stops being blended and runs away unguided. Not a warmup or shape issue —
  graphs are incompatible with this CFG implementation. Reverted.
* **`torch.compile(mode="reduce-overhead")` on the flow estimator**: 1.36x at
  1 row, **1.03x at 4 rows**. At batch >= 4 the acoustic decode is genuinely
  compute-bound, not launch-bound, so there is nothing for graphs to remove.
* **Shorter acoustic prompt**: 250 → 50 codes gives only **1.4x** at 8 rows
  (50.5 → 35.6 ms/row) and *nothing* at 1 row. The earlier finding that prompt
  length does not drive cost holds in both regimes.
* **Batching saturates at 4 rows**: 110.9 → 51.8 ms/row, then flat (~49 ms at 8,
  16 and 32 rows). Grouping more concurrent streams buys nothing past 4.

### A crash the "obvious" floor would have shipped

Setting the first rung to 4 (lookahead + 1, which *looks* legal) killed the
engine under load: **952 failures, zero turns**. The binding constraint is the
**HiFT vocoder**, not the encoder lookahead — it reflection-pads 1024 samples
per side, so one emitted code (960 samples) makes `pad` raise *"padding size
should be less than the corresponding input dimension"*. Two codes clear it, so
the true floor for the first rung is **5**. Now a no-op instead of a crash, with
`ACOUSTIC_MIN_EMIT_CODES` and two gates pinning it.

### Why sub-1 s at 20–30 callers is not reachable on this box

With quality held fixed the arithmetic closes out:

```
TTFA(20 callers) ~= prefill 150 ms + 5 codes x 85 ms + acoustic 110 ms + ~250 ms fixed
                 ~= 1.2 s  (measured p50 1.21 s)
```

Getting p95 under 1 s at 20 callers needs stage-0 ITL near 30–40 ms *under
load*, i.e. roughly half the total GPU work — which is not available with F32,
CFG (two sequences per request) and eager execution all held. **p95 under 1 s is
delivered at 8 concurrent calls; p50 under 1 s up to 16.**

Three routes past it, in order of value:

1. **A second GPU.** Stage 0 and stage 1 currently contend for one card; the
   acoustic stage takes roughly half the GPU. Splitting the stages removes the
   5.5–9x ITL inflation directly. This box has one RTX 5090, so it could not be
   measured here.
2. **Make CFG CUDA-graph compatible** (quality-neutral). Stage-0 steps are
   launch-bound at 15–17 ms idle for a 520 M model; graphs are the natural fix
   and are blocked only by the `_sample` pairing hook, not by anything
   fundamental.
3. **BF16.** Excluded here as a quality change; it is next in the plan's own
   optimization order.
