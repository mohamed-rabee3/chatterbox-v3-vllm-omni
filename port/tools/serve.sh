#!/bin/bash
# Run from a full vLLM-Omni checkout with this overlay installed. A namespace
# import from the overlay directory alone hides upstream runtime modules.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "${VENV:-/venv/main}/bin/activate"
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export TORCH_DISABLE_NATIVE_JIT=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export PYTHONUNBUFFERED=1
# Old CUDA-12.8 hosts may need an explicit TRITON_PTXAS_BLACKWELL_PATH to
# Triton's CUDA-12 ptxas. Do not force that workaround on newer driver stacks.
MODEL=${MODEL:-/workspace/models/chatterbox-mtl-v3}
PORT=${PORT:-18091}
HOST=${HOST:-127.0.0.1}
LOG=${LOG:-$ROOT/port/artifacts/server.log}
cd "${OMNI_ROOT:-/workspace/repos/vllm-omni}"
if [[ "$LOG" != "-" ]]; then
  mkdir -p "$(dirname -- "$LOG")"
  exec >"$LOG" 2>&1
fi
deploy_args=(--deploy-config "${DEPLOY_CONFIG:-$ROOT/vllm_omni/deploy/chatterbox_mtl_v3_low_latency.yaml}")
CONFIG_PATH=${DEPLOY_CONFIG:-$ROOT/vllm_omni/deploy/chatterbox_mtl_v3_low_latency.yaml}
for arg in "$@"; do
  case "$arg" in
    --deploy-config=*) deploy_args=(); CONFIG_PATH=${arg#--deploy-config=} ;;
    --deploy-config) deploy_args=(); NEXT_IS_CONFIG=1 ;;
    *) if [[ "${NEXT_IS_CONFIG:-}" == 1 ]]; then CONFIG_PATH=$arg; NEXT_IS_CONFIG=; fi ;;
  esac
done

# The streaming chunk ladder must be identical in BOTH stage processes: stage 0
# transports codec blocks on it and stage 1 decodes on it, and the connector
# rejects the deploy if they disagree. They read different sections of this file
# and run as separate processes, so the profile's connector block is treated as
# the single source of truth and exported to the environment both inherit.
# Without this, a profile carrying a non-default ladder would start under
# supervisor (where the values are set by hand) but fail from this script.
eval "$(python - "$CONFIG_PATH" <<'PY'
import sys, yaml
try:
    with open(sys.argv[1]) as fh:
        extra = ((yaml.safe_load(fh) or {}).get("connectors") or {}).get(
            "connector_of_shared_memory", {}).get("extra", {}) or {}
except Exception:
    extra = {}
for env, key in (("CBX_STREAM_FIRST_BLOCK", "codec_chunk_frames"),
                 ("CBX_STREAM_GROWTH", "codec_chunk_growth"),
                 ("CBX_STREAM_MAX_BLOCK", "codec_max_chunk_frames")):
    if key in extra:
        print(f"export {env}={extra[key]}")
PY
)"
exec vllm serve "$MODEL" --omni --host "$HOST" --port "$PORT" "${deploy_args[@]}" "$@"
