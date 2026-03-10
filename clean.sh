#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "Cleaning build artifacts..."

# Top-level release tarball
rm -f dress-graph-*.tar.gz

# CMake build directory
rm -rf build/

# C/C++ objects and libraries (everywhere)
find . -name '*.o' -not -path './.git/*' -not -path './.venv/*' -delete
find . -name '*.a' -not -path './.git/*' -not -path './.venv/*' -delete
find . -name '*.so' -not -path './.git/*' -not -path './.venv/*' -delete
find . -name '*.dylib' -not -path './.git/*' -not -path './.venv/*' -delete

# CMake generated files (in-source)
rm -f CMakeCache.txt cmake_install.cmake Makefile
rm -rf CMakeFiles/

# C/C++ test binaries
rm -f tests/c/test_dress
rm -f tests/c/test_dress_igraph
rm -f tests/c/test_delta_dress
rm -f tests/c/test_delta_dress_igraph
rm -f tests/cpp/test_dress
rm -f tests/cpp/test_delta_dress

# C/C++ example binaries
rm -f examples/c/cpu examples/c/cuda examples/c/mpi examples/c/mpi_cuda
rm -f examples/c/cpu_igraph examples/c/cuda_igraph
rm -f examples/c/mpi_igraph examples/c/mpi_cuda_igraph
rm -f examples/cpp/cpu examples/cpp/cuda examples/cpp/mpi examples/cpp/mpi_cuda

# Rust
rm -rf rust/target/
rm -rf rust/vendor/
rm -rf tests/rust/target/
rm -rf examples/rust/target/
find . -name 'Cargo.lock' -not -path './.git/*' -delete

# Python
rm -rf python/build/
rm -rf python/dist/
rm -rf python/src/*.egg-info/
rm -rf python/src/dress/*.so
rm -f  python/_dress.c
rm -f  python/_delta_dress.c
rm -f  python/_delta_dress_impl.c
rm -f  python/delta_dress_impl.h
find . -name '__pycache__' -not -path './.git/*' -exec rm -rf {} +
find . -name '*.pyc'       -not -path './.git/*' -delete

# WASM (Emscripten output)
rm -f wasm/dress_wasm.cjs
rm -f wasm/dress_wasm.wasm
rm -f wasm/dress-graph-*.tgz

# WASM example install artifacts
rm -rf examples/wasm/node_modules/
rm -f  examples/wasm/package-lock.json
rm -f  examples/wasm/package.json

# Emscripten SDK (downloaded by build)
rm -rf emsdk/

# MkDocs
rm -rf site/

# Go
rm -f tests/go/go.sum

# Matlab / Octave
find . -name '*.mex' -not -path './.git/*' -delete
find . -name '*.oct' -not -path './.git/*' -delete

# R
rm -rf dress.graph.Rcheck/
rm -rf dress.Rcheck/
rm -f  dress.graph_*.tar.gz

# LaTeX auxiliary files (research/)
find research/ -type f \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.blg' \) -delete 2>/dev/null || true

# pytest
rm -rf .pytest_cache/
find . -name '.pytest_cache' -not -path './.git/*' -exec rm -rf {} +

echo "Done."
