#!/usr/bin/env bash
# tqLLM 1.4.4 — Qwen3.8 Flash-Next mixed tqfn, 32K ctx, thinking medium, MTP drafter
set -e
docker rm -f tqfn 2>/dev/null || true
docker run -d --gpus all -p 192.168.100.11:8086:8086 \
  -v /home/leilei/models/tqllm/flashnext:/fn:ro \
  --entrypoint tqserve --name tqfn ytan2000/tqllm:1.4.4 \
  --model /fn/flashnext-tqllm.tqfn \
  --tokenizer /fn/flashnext-tokenizer.gguf \
  --context 524288 --np 2 --prompt-cache on \
  --host 0.0.0.0 --port 8086 \
  --reasoning on --reasoning-effort medium --max-tokens 16384 \
  --name tqfn
