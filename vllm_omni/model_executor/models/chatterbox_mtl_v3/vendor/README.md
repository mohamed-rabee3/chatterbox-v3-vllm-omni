# Vendored Chatterbox model modules

Verbatim copy of `src/chatterbox/models/` from
https://github.com/resemble-ai/chatterbox at pinned revision
`5de7a54aa4e5e2baadb0182dde554908b48b85c2`, under the MIT License (see
`LICENSE`, retained unchanged).

Why vendored rather than depended on:

* the upstream package pins `torch==2.6.0` / `transformers==5.2.0`; resolving
  those into the vLLM-Omni environment would replace the engine's own
  Torch/Transformers build (port plan section 2.1);
* serving requires removing process-global state and making the acoustic
  stochastic inputs request-scoped, which means these modules must be editable.

**Every divergence from upstream is marked with a `# PORT:` comment.** A file
with no `# PORT:` comment is byte-identical to upstream. Run
`python /workspace/port/tools/check_vendor_drift.py` to verify that claim.
