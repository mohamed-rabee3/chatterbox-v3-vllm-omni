#!/bin/bash
# Locust conversation sweep. Set LOAD_WAIT_MIN=0 LOAD_WAIT_MAX=0 for
# continuously active synthesis requests instead of conversations.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
BASE=${BASE:-http://127.0.0.1:18091}
DUR=${DUR:-3m}
OUT=${OUT:-$ROOT/port/artifacts/sweep}
PYTHON=${PYTHON:-/venv/main/bin/python}
LOCUST=${LOCUST:-/venv/main/bin/locust}
mkdir -p "$OUT"
STATUS=0
for USERS in ${LEVELS:-1 20 30}; do
  mkdir -p "$OUT/u$USERS"
  if LOAD_ARTIFACTS="$OUT/u$USERS" "$LOCUST" \
      -f "$ROOT/port/tools/locustfile_conversation.py" --headless \
      -u "$USERS" -r "${SPAWN_RATE:-5}" -t "$DUR" \
      --stop-timeout "${DRAIN_SECONDS:-90}" --host "$BASE" --only-summary \
      > "$OUT/u$USERS/locust.log" 2>&1; then
    RUN_STATUS=0
  else
    RUN_STATUS=$?
    STATUS=1
    echo "users=$USERS Locust exit=$RUN_STATUS; retaining failure results"
  fi
  "$PYTHON" - "$OUT/u$USERS/locust_conversation.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
t = d['ttfa_s']
print(f"users={d['users']} active_peak={d['peak_inflight_requests']} "
      f"completed={d['turns_completed']} failures={d['failures']} "
      f"unfinished={d['requests_unfinished']} "
      f"TTFA p50={t['p50']:.3f}s p95={t['p95']:.3f}s "
      f"playback_stall_p95={d['playback_stall_s_p95']:.3f}s "
      f"turns/s={d['turns_per_s']:.2f}")
PY
done
exit "$STATUS"
