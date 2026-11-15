#!/usr/bin/env bash
# Build the llama.cpp runtime for the llm-cpp device.
# Produces fully static binaries (no glibc/version dependency) so they run on
# any Linux x86_64 host, including Vercel's older runtime.
set -euo pipefail

LLAMA_REF="${LLAMA_REF:-master}"
WORK="${WORK:-/tmp/llama.cpp}"
OUT="$(cd "$(dirname "$0")/.." && pwd)/bin"

command -v cmake >/dev/null || { echo "cmake required"; exit 1; }

if [ ! -d "$WORK/.git" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$WORK"
fi
cd "$WORK"
git fetch --depth 1 origin "$LLAMA_REF" && git checkout FETCH_HEAD

cmake -B build-static \
  -DGGML_NATIVE=OFF \
  -DGGML_STATIC=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS="-static"

cmake --build build-static --config Release -j"$(nproc)" --target llama-cli llama-server

mkdir -p "$OUT"
cp build-static/bin/llama-cli build-static/bin/llama-server "$OUT/"
echo "Built static binaries:"
ls -lh "$OUT"
