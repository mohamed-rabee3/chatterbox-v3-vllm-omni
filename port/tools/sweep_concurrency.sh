#!/bin/bash
# Conversational capacity sweep: find the highest number of concurrent CALLS
# that still meets a time-to-first-audio budget. Each level is a fresh locust
# run against the live server; results land in their own directory so nothing
# overwrites anything.
set -u
BASE=${BASE:-http://127.0.0.1:18091}
DUR=${DUR:-3m}
OUT=/workspace/port/artifacts/sweep
mkdir -p "$OUT"
for U in ${LEVELS:-8 16 30 48}; do
  echo "=== $U concurrent conversations ==="
  mkdir -p "$OUT/u$U"
  LOAD_ARTIFACTS="$OUT/u$U" /venv/main/bin/locust \
      -f /workspace/port/tools/locustfile_conversation.py --headless \
      -u "$U" -r 4 -t "$DUR" --host "$BASE" --only-summary \
      > "$OUT/u$U/locust.log" 2>&1
  /venv/main/bin/python - "$U" <<'PY'
import json, sys
u = sys.argv[1]
d = json.load(open(f"/workspace/port/artifacts/sweep/u{u}/locust_conversation.json"))
t = d["ttfa_s"]
print(f"  users={u:>3} turns={d['turns_completed']:>4} fail={d['failures']:>3} "
      f"TTFA p50={t['p50']:6.2f}s p95={t['p95']:6.2f}s p99={t['p99']:6.2f}s  "
      f"turns/s={d['turns_per_s']:.2f}  audio_s/s={d['audio_s_per_wall_s']:.2f}")
PY
done
echo SWEEP_DONE
