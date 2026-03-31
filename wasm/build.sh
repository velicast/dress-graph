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

EXPORTED_FUNCTIONS='["_dress_init_graph","_dress_fit","_dress_get","_dress_free_graph","_dress_delta_fit","_dress_delta_fit_strided","_dress_nabla_fit","_malloc","_free"]'
EXPORTED_RUNTIME='["ccall","cwrap","getValue","setValue","HEAP32","HEAPU32","HEAPF64"]'

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
    "$LIB_DIR/src/delta_dress.c" \
    "$LIB_DIR/src/delta_dress_impl.c" \
    "$LIB_DIR/src/nabla_dress.c" \
    "$LIB_DIR/src/nabla_dress_impl.c" \
    "$LIB_DIR/src/dress_histogram.c" \
    -I "$LIB_DIR/include" \
    -o "$OUT_DIR/dress_wasm.cjs"

echo "Output:"
ls -lh "$OUT_DIR/dress_wasm.cjs" "$OUT_DIR/dress_wasm.wasm" 2>/dev/null || true
echo "Done."
