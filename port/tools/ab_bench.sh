#!/bin/bash
# Wait for the server, then run the concurrency levels used for the A/B.
set -uo pipefail
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home TORCH_DISABLE_NATIVE_JIT=1
export TRITON_PTXAS_BLACKWELL_PATH=/venv/main/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
cd /workspace/port
OUT=${OUT:-/workspace/port/artifacts/bench_2x.json}
LOG=${ABLOG:-/workspace/port/artifacts/ab.log}
: > "$LOG"
for i in $(seq 1 240); do
  if [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:18091/health 2>/dev/null)" = "200" ]; then
    echo "SERVER READY after $((i*10))s" | tee -a "$LOG"; break
  fi
  pgrep -f "vllm serve" >/dev/null || { echo "SERVER DIED" | tee -a "$LOG"; exit 1; }
  sleep 10
done
python tools/bench.py --concurrency 8,16,32 --out "$OUT" 2>&1 | grep -E '^\{|^==|^  ' | tee -a "$LOG"
echo "AB DONE" | tee -a "$LOG"
