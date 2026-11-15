#!/usr/bin/env bash
# Pull a model from HuggingFace by repo id + filename, then serve it with llama-server.
# Usage: ./run-device.sh <hf-repo-id> <gguf-filename> [port]
# Example: ./run-device.sh Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf
set -euo pipefail

HF_REPO="${1:?usage: run-device.sh <hf-repo-id> <gguf-file> [port]}"
HF_FILE="${2:?usage: run-device.sh <hf-repo-id> <gguf-file> [port]}"
PORT="${3:-18080}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$ROOT/models/$(basename "$HF_FILE")"
URL="https://huggingface.co/$HF_REPO/resolve/main/$HF_FILE"

if [ ! -f "$MODEL" ]; then
  echo "Pulling $HF_REPO/$HF_FILE from HuggingFace ..."
  curl -fL --retry 3 -o "$MODEL" "$URL"
fi
echo "Model ready: $(ls -lh "$MODEL" | awk '{print $5}') $MODEL"

exec "$ROOT/bin/llama-server" \
  -m "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  -t "$(nproc)" -c 4096
