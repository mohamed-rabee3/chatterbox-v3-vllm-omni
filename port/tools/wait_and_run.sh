#!/bin/bash
# Wait for the server to become healthy, then run a command.
for i in $(seq 1 90); do
  if [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:18091/health 2>/dev/null)" = "200" ]; then
    echo "SERVER READY after $((i*5))s"
    exec "$@"
  fi
  pgrep -f "vllm serve" >/dev/null || { echo "SERVER DIED"; exit 1; }
  sleep 5
done
echo "TIMEOUT waiting for server"
exit 1
