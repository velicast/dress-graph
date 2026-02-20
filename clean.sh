#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "Cleaning build artifacts..."

# CMake build directory
rm -rf build/

# C/C++ objects and libraries
find . -name '*.o' -not -path './.git/*' -delete
find . -name '*.a' -not -path './.git/*' -delete
find . -name '*.so' -not -path './.git/*' -delete
find . -name '*.dylib' -not -path './.git/*' -delete

# CMake generated files (in-source)
rm -f CMakeCache.txt cmake_install.cmake Makefile
rm -rf CMakeFiles/

# C/C++ test binaries
rm -f tests/c/test_dress
rm -f tests/cpp/test_dress

# Rust
rm -rf rust/target/
rm -rf tests/rust/target/

# Python
rm -rf python/build/
rm -rf python/src/*.egg-info/
rm -rf python/src/dress/*.so
rm -f  python/_dress.c
find . -name '__pycache__' -not -path './.git/*' -exec rm -rf {} +
find . -name '*.pyc'       -not -path './.git/*' -delete

# WASM (Emscripten output)
rm -f wasm/dress_wasm.cjs
rm -f wasm/dress_wasm.wasm

# Emscripten SDK (downloaded by build)
rm -rf emsdk/

# MkDocs
rm -rf site/

# Go
rm -f tests/go/go.sum

# R
rm -rf dress.graph.Rcheck/
rm -rf dress.Rcheck/
rm -f  dress.graph_*.tar.gz

# pytest
rm -rf .pytest_cache/
find . -name '.pytest_cache' -not -path './.git/*' -exec rm -rf {} +

echo "Done."
