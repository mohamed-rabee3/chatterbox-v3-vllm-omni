#!/bin/bash
# Launch the Chatterbox Multilingual V3 vLLM-Omni server.
#
# Two environment fixes are REQUIRED on this stack (RTX 5090 / sm_120,
# driver 570.211.01 = CUDA 12.8, torch 2.13+cu129, triton 3.7.1):
#
# 1. TRITON_PTXAS_BLACKWELL_PATH -- Triton routes every sm >= 100 kernel through
#    its bundled `ptxas-blackwell`, which is CUDA **13.1**. A CUDA 13.x cubin
#    cannot be loaded by a CUDA 12.8 driver, so EVERY Triton kernel failed with
#    "Triton Error [CUDA]: device kernel image is invalid" -- including the
#    TRITON_ATTN attention backend and vLLM's top-k/top-p sampler. Triton's
#    other bundled `ptxas` is CUDA 12.8 and does support sm_120, so pointing the
#    Blackwell slot at it makes Triton work. Verified with a minimal kernel.
#
# 2. TORCH_DISABLE_NATIVE_JIT=1 -- torch 2.13's `torch._native` eager router
#    replaces matmul with its own Triton kernel. Redundant once (1) is set, but
#    kept so the reference runner and the tests behave identically whether or
#    not (1) is exported.
set -euo pipefail
source /venv/main/bin/activate
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export TRITON_PTXAS_BLACKWELL_PATH=${TRITON_PTXAS_BLACKWELL_PATH:-/venv/main/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas}
export TORCH_DISABLE_NATIVE_JIT=1
# FlashInfer's JIT refuses to build for this GPU ("FlashInfer requires GPUs
# with sm75 or higher" is its message for an sm_120 part it cannot handle).
# vLLM's native PyTorch sampler is the reference-order path this model's
# fidelity gates were measured against anyway.
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
MODEL=${MODEL:-/workspace/models/chatterbox-mtl-v3}
PORT=${PORT:-18091}
LOG=${LOG:-/workspace/port/artifacts/server.log}
exec vllm serve "$MODEL" --omni --port "$PORT" "$@" >"$LOG" 2>&1
