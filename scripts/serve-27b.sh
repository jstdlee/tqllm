#!/usr/bin/env bash
# tqLLM 1.4.4 — Qwen3.8-27B TQ_L, 32K ctx, thinking medium, DFlash2 drafter
set -e
docker rm -f tq27b 2>/dev/null || true
docker run -d --gpus all -p 127.0.0.1:8086:8086 \
  -v /home/leilei/models/tqllm/27b:/models:ro \
  --entrypoint tqserve --name tq27b ytan2000/tqllm:1.4.4 \
  --model /models/Qwen3.8-27B-TQ_L.gguf \
  -md /models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf \
  --draft-vocab /models/mtp-vocab/atx_65536.txt \
  --context 262144 --host 0.0.0.0 --port 8086 \
  --reasoning on --reasoning-effort medium --max-tokens 16384 \
  --name tq27b
