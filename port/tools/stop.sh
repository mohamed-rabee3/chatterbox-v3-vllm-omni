#!/bin/bash
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "StageEngineCore" 2>/dev/null || true
sleep 3
exit 0
