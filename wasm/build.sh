#!/usr/bin/env bash
# build.sh — Compile dress.c to WebAssembly via Emscripten.
#
# Prerequisites:
#   source /path/to/emsdk/emsdk_env.sh
#
# Usage:
#   ./build.sh            # default optimised build
#   ./build.sh --debug    # debug build with assertions
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIB_DIR="$SCRIPT_DIR/../libdress"
OUT_DIR="$SCRIPT_DIR"

DEBUG=0
if [[ "${1:-}" == "--debug" ]]; then
    DEBUG=1
fi

if ! command -v emcc &>/dev/null; then
    echo "ERROR: emcc not found. Install Emscripten and source emsdk_env.sh first."
    echo "  https://emscripten.org/docs/getting_started/downloads.html"
    exit 1
fi

EXPORTED_FUNCTIONS='["_init_dress_graph","_fit","_free_dress_graph","_malloc","_free"]'
EXPORTED_RUNTIME='["ccall","cwrap","getValue","setValue","HEAP32","HEAPF64"]'

FLAGS=(
    -s EXPORTED_FUNCTIONS="$EXPORTED_FUNCTIONS"
    -s EXPORTED_RUNTIME_METHODS="$EXPORTED_RUNTIME"
    -s ALLOW_MEMORY_GROWTH=1
    -s MODULARIZE=1
    -s EXPORT_NAME="createDressModule"
    -s ENVIRONMENT="web,node"
    -s FILESYSTEM=0
    -s SINGLE_FILE=0
    -lm
)

if [[ $DEBUG -eq 1 ]]; then
    FLAGS+=(-O0 -g -s ASSERTIONS=2 -s SAFE_HEAP=1)
    echo "Building debug…"
else
    FLAGS+=(-O3 -DNDEBUG)
    echo "Building release…"
fi

emcc "${FLAGS[@]}" \
    "$LIB_DIR/src/dress.c" \
    -I "$LIB_DIR/include" \
    -o "$OUT_DIR/dress_wasm.cjs"

echo "Output:"
ls -lh "$OUT_DIR/dress_wasm.cjs" "$OUT_DIR/dress_wasm.wasm" 2>/dev/null || true
echo "Done."
