#!/bin/bash
# Wait for the server, then run the full serving verification suite.
# Everything lands in /workspace/port/artifacts/suite.log so results survive
# the shell that started them.
set -uo pipefail
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export TORCH_DISABLE_NATIVE_JIT=1
export TRITON_PTXAS_BLACKWELL_PATH=/venv/main/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
cd /workspace/port
LOG=/workspace/port/artifacts/suite.log
: > "$LOG"

for i in $(seq 1 120); do
  if [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:18091/health 2>/dev/null)" = "200" ]; then
    echo "SERVER READY after $((i*5))s" | tee -a "$LOG"; break
  fi
  pgrep -f "vllm serve" >/dev/null || { echo "SERVER DIED" | tee -a "$LOG"; exit 1; }
  sleep 5
done

for step in "$@"; do
  echo "" | tee -a "$LOG"
  echo "########## $step ##########" | tee -a "$LOG"
  case "$step" in
    smoke)  python tools/smoke_client.py 2>&1 | grep -vE "^\s*$" | tee -a "$LOG" ;;
    verify) python tools/verify_audio.py 2>&1 | grep -E "^\{|^==|^ *case|^[a-z_]+ +(en|ar) " | tee -a "$LOG" ;;
    bench)  python tools/bench.py 2>&1 | grep -E "^\{|^==|^  " | tee -a "$LOG" ;;
    conc)   python tools/test_concurrency.py 2>&1 | grep -E "^\{|^==|^  " | tee -a "$LOG" ;;
    seeds)  python tools/diag_seeds.py 2>&1 | grep -E "^\{|^==|^[a-z_]+ +[0-9]" | tee -a "$LOG" ;;
    soak)   python tools/bench.py --concurrency "" --open-loop-rate 1.2 --open-loop-seconds 240 \
              --out /workspace/port/artifacts/soak.json 2>&1 | grep -E "^\{|^==|^  " | tee -a "$LOG" ;;
    *)      echo "unknown step $step" | tee -a "$LOG" ;;
  esac
done
echo "" | tee -a "$LOG"
echo "SUITE DONE" | tee -a "$LOG"
