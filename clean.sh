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

# CUDA build artifacts (in-tree Makefile build)
# Keep libdress/src/cuda/Makefile — it is a source file, not a build artifact.
rm -f  libdress/src/cuda/test_dress_cuda
rm -f  libdress/src/cuda/libdress_cuda.a
rm -f  libdress/src/cuda/libdress_cuda.so
# .o files already cleaned by the global find above

# C/C++ example binaries
rm -f examples/c/cpu examples/c/cuda examples/c/mpi examples/c/mpi_cuda
rm -f examples/c/cpu_igraph examples/c/cuda_igraph
rm -f examples/c/mpi_igraph examples/c/mpi_cuda_igraph
rm -f examples/cpp/cpu examples/cpp/cuda examples/cpp/mpi examples/cpp/mpi_cuda

# Rust
rm -rf rust/target/
rm -rf rust/vendor/
rm -f  rust/LICENSE
rm -rf tests/rust/target/
rm -rf examples/rust/target/
find . -name 'Cargo.lock' -not -path './.git/*' -delete

# Python
rm -rf python/build/
rm -rf python/dist/
rm -rf python/src/*.egg-info/
rm -rf python/src/dress_graph.egg-info/
rm -rf python/src/dress/*.so
rm -rf python/src/dress/_vendored/
rm -f  python/_dress.c
rm -f  python/_delta_dress.c
rm -f  python/_delta_dress_impl.c
rm -f  python/delta_dress_impl.h
find . -name '__pycache__' -not -path './.git/*' -not -path './.venv/*' -exec rm -rf {} +
find . -name '*.pyc'       -not -path './.git/*' -not -path './.venv/*' -delete

# WASM (Emscripten output)
rm -f wasm/dress_wasm.cjs
rm -f wasm/dress_wasm.wasm
rm -f wasm/dress-graph-*.tgz
rm -f wasm/.npmrc

# WASM example install artifacts
rm -rf examples/wasm/node_modules/
rm -f  examples/wasm/package-lock.json
rm -f  examples/wasm/package.json

# Emscripten SDK (downloaded by build)
rm -rf emsdk/

# MkDocs
rm -rf site/

# Go (vendored C sources, created by publish.sh / run_examples.sh)
rm -rf go/vendor/
rm -rf go/cuda/vendor/
rm -rf go/mpi/vendor/
rm -rf go/mpi/cuda/vendor/
rm -f  tests/go/go.sum

# Julia (vendored C sources and compiled shared library)
rm -rf julia/vendor/
rm -f  julia/libdress.so

# Matlab / Octave
find . -name '*.mex' -not -path './.git/*' -not -path './.venv/*' -delete
find . -name '*.oct' -not -path './.git/*' -not -path './.venv/*' -delete
rm -rf octave/inst/
# Keep octave/src/Makefile (source file), remove only build outputs
find octave/src/ -type f ! -name 'Makefile' -delete 2>/dev/null || true

# R
rm -rf dress.graph.Rcheck/
rm -rf dress.Rcheck/
rm -f  dress.graph_*.tar.gz
# Vendored C sources copied into r/src/ at publish time
rm -f  r/src/dress.c r/src/delta_dress.c r/src/delta_dress_impl.c r/src/delta_dress_impl.h
rm -f  r/src/dress_mpi.c r/src/delta_dress_cuda.c r/src/dress_cuda.cu r/src/dress_cuda.o
rm -rf r/src/dress/

# LaTeX auxiliary files (research/)
find research/ -type f \( -name '*.aux' -o -name '*.log' -o -name '*.out' \
    -o -name '*.blg' -o -name '*.bbl' -o -name '*.fls' -o -name '*.fdb_latexmk' \
    -o -name '*.synctex.gz' -o -name '*.toc' \) -delete 2>/dev/null || true

# pytest
rm -rf .pytest_cache/
find . -name '.pytest_cache' -not -path './.git/*' -exec rm -rf {} +

echo "Done."
