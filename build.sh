#!/usr/bin/env bash
# build.sh — Build and test every language binding in the dress-graph repo.
#
# Usage:
#   ./build.sh            # build + test everything available
#   ./build.sh --no-test  # build only, skip tests
#   ./build.sh c cpp      # build + test only the listed targets
#
# Targets: c cpp igraph python rust go r julia wasm octave
#
# A target is silently skipped when its toolchain is not installed.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── source emsdk if available and emcc is not already on PATH ───────
if ! command -v emcc &>/dev/null && [[ -f "$HOME/emsdk/emsdk_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/emsdk/emsdk_env.sh" 2>/dev/null
fi

# ── parse flags ─────────────────────────────────────────────────────
RUN_TESTS=1
TARGETS=()
for arg in "$@"; do
    case "$arg" in
        --no-test) RUN_TESTS=0 ;;
        *)         TARGETS+=("$arg") ;;
    esac
done

# Default: all targets
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(c cpp igraph python rust go r julia wasm octave)
fi

PASS=0
FAIL=0
SKIP=0

pass()  { echo "  ✓ $1"; PASS=$((PASS + 1)); }
fail()  { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }
skip()  { echo "  – $1 (skipped, toolchain not found)"; SKIP=$((SKIP + 1)); }
header(){ echo; echo "── $1 ──────────────────────────────────────────"; }

# Run a command; on success call pass, on failure call fail.
run_step() {
    local label="$1"; shift
    if "$@"; then
        pass "$label"
    else
        fail "$label"
    fi
}
header(){ echo; echo "── $1 ──────────────────────────────────────────"; }

want() { [[ " ${TARGETS[*]} " == *" $1 "* ]]; }

# Copy libdress sources into Rust and R vendor directories.
vendor_sources() {
    # Rust
    mkdir -p rust/vendor/include/dress/cuda
    cp libdress/src/dress.c            rust/vendor/dress.c
    cp libdress/src/delta_dress.c      rust/vendor/delta_dress.c
    cp libdress/src/delta_dress_impl.c rust/vendor/delta_dress_impl.c
    cp libdress/src/delta_dress_impl.h rust/vendor/delta_dress_impl.h
    cp libdress/include/dress/dress.h       rust/vendor/include/dress/dress.h
    cp libdress/include/dress/delta_dress.h rust/vendor/include/dress/delta_dress.h
    cp libdress/include/dress/cuda/dress_cuda.h rust/vendor/include/dress/cuda/dress_cuda.h
    cp libdress/src/cuda/delta_dress_cuda.c     rust/vendor/delta_dress_cuda.c

    # R
    mkdir -p r/src/dress/cuda r/src/dress/mpi
    cp libdress/src/dress.c            r/src/dress.c
    cp libdress/src/delta_dress.c      r/src/delta_dress.c
    cp libdress/src/delta_dress_impl.c r/src/delta_dress_impl.c
    cp libdress/src/delta_dress_impl.h r/src/delta_dress_impl.h
    cp libdress/src/cuda/delta_dress_cuda.c   r/src/delta_dress_cuda.c
    cp libdress/src/cuda/dress_cuda.cu        r/src/dress_cuda.cu
    cp libdress/include/dress/dress.h            r/src/dress/dress.h
    cp libdress/include/dress/delta_dress.h      r/src/dress/delta_dress.h
    cp libdress/include/dress/cuda/dress_cuda.h  r/src/dress/cuda/dress_cuda.h
    cp libdress/include/dress/mpi/dress_mpi.h    r/src/dress/mpi/dress_mpi.h
    cp libdress/src/mpi/dress_mpi.c              r/src/dress_mpi.c
}

# Remove vendored copies.
unvendor_sources() {
    rm -rf rust/vendor
    rm -f  r/src/dress.c r/src/delta_dress.c r/src/delta_dress_impl.c r/src/delta_dress_impl.h
    rm -f  r/src/delta_dress_cuda.c r/src/dress_cuda.cu r/src/dress_cuda.o r/src/dress_mpi.c
    rm -f  r/src/Makevars
    rm -rf r/src/dress
    rm -f  python/_dress.c python/_delta_dress.c python/_delta_dress_impl.c python/delta_dress_impl.h
}

# ── C / C++ (CMake) ────────────────────────────────────────────────
build_c_cpp() {
    if ! command -v cmake &>/dev/null; then
        want c   && skip "C (cmake not found)"
        want cpp && skip "C++ (cmake not found)"
        return
    fi

    header "C / C++ (CMake)"
    mkdir -p build

    # Auto-detect MPI
    local CMAKE_EXTRA=""
    if command -v mpicc &>/dev/null; then
        CMAKE_EXTRA="-DDRESS_MPI=ON"
    fi

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release $CMAKE_EXTRA 2>&1 | tail -1
    cmake --build build -j"$(nproc 2>/dev/null || echo 2)" 2>&1
    pass "libdress built"

    # Build CUDA library (shared + static) if nvcc is available
    if command -v nvcc &>/dev/null && [[ -f libdress/src/cuda/Makefile ]]; then
        make -C libdress/src/cuda clean all 2>&1 | tail -3
        cp libdress/src/cuda/libdress_cuda.so build/libdress/
        cp libdress/src/cuda/libdress_cuda.a  build/libdress/

        # Vendor pre-compiled CUDA kernel object into each wrapper
        cp libdress/src/cuda/dress_cuda.o rust/vendor/dress_cuda.o
        pass "libdress_cuda built"
    fi

    if [[ $RUN_TESTS -eq 1 ]]; then
        # Extra link flags when CUDA libs are present (prefer static archive)
        local CUDA_LIBS=""
        if [[ -f build/libdress/libdress_cuda.a ]]; then
            CUDA_LIBS="build/libdress/libdress_cuda.a -lcudart_static -ldl -lrt -lpthread"
        fi

        # C tests
        if want c && [[ -f tests/c/test_dress.c ]]; then
            gcc -O2 -o tests/c/test_dress tests/c/test_dress.c \
                -Ilibdress/include build/libdress/libdress.a $CUDA_LIBS -lm -fopenmp 2>&1
            run_step "C tests" tests/c/test_dress
        fi

        # C++ tests
        if want cpp && [[ -f tests/cpp/test_dress.cpp ]]; then
            g++ -O2 -std=c++17 -o tests/cpp/test_dress tests/cpp/test_dress.cpp \
                -Ilibdress/include -Ilibdress++/include \
                build/libdress/libdress.a $CUDA_LIBS -lm -fopenmp 2>&1
            run_step "C++ tests" tests/cpp/test_dress
        fi
    fi
}

# ── C igraph wrapper ────────────────────────────────────────────────
build_igraph() {
    want igraph || return 0
    if ! pkg-config --exists igraph 2>/dev/null; then
        skip "igraph (pkg-config igraph not found)"; return
    fi
    header "C igraph wrapper"

    local IGRAPH_CFLAGS IGRAPH_LIBS
    IGRAPH_CFLAGS=$(pkg-config --cflags igraph)
    IGRAPH_LIBS=$(pkg-config --libs igraph)

    # Build + test: DRESS igraph wrapper
    if [[ -f tests/c/test_dress_igraph.c ]]; then
        gcc -O2 -I libdress/include -I libdress-igraph/include \
            $IGRAPH_CFLAGS \
            -o tests/c/test_dress_igraph \
            tests/c/test_dress_igraph.c \
            libdress-igraph/src/dress_igraph.c \
            libdress/src/dress.c \
            libdress/src/delta_dress.c \
            libdress/src/delta_dress_impl.c \
            $IGRAPH_LIBS -lm -fopenmp 2>&1
        pass "libdress-igraph compiled"

        if [[ $RUN_TESTS -eq 1 ]]; then
            run_step "igraph DRESS tests" tests/c/test_dress_igraph
        fi
    fi

    # Build + test: Δ^k-DRESS igraph wrapper
    if [[ -f tests/c/test_delta_dress_igraph.c ]]; then
        gcc -O2 -I libdress/include -I libdress-igraph/include \
            $IGRAPH_CFLAGS \
            -o tests/c/test_delta_dress_igraph \
            tests/c/test_delta_dress_igraph.c \
            libdress-igraph/src/dress_igraph.c \
            libdress/src/dress.c \
            libdress/src/delta_dress.c \
            libdress/src/delta_dress_impl.c \
            $IGRAPH_LIBS -lm -fopenmp 2>&1
        pass "libdress-igraph delta compiled"

        if [[ $RUN_TESTS -eq 1 ]]; then
            run_step "igraph Δ^k-DRESS tests" tests/c/test_delta_dress_igraph
        fi
    fi
}

# ── Python ──────────────────────────────────────────────────────────
build_python() {
    want python || return 0
    if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
        skip "Python"; return
    fi
    header "Python"
    local PY
    PY=$(command -v python3 || command -v python)

    # Prefer the repo venv if it exists
    if [[ -f "$ROOT/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$ROOT/.venv/bin/activate"
        PY=python
    fi

    # Ensure pybind11 + numpy are available for the native C extension
    $PY -m pip install pybind11 numpy --quiet 2>&1

    DRESS_BUILD_NATIVE=1 $PY -m pip install -e "python/[native]" --no-build-isolation --quiet 2>&1
    pass "Python package installed"

    # Verify the C extension was built
    if $PY -c "import dress; assert dress._BACKEND == 'c'" 2>/dev/null; then
        pass "pybind11 C extension active"
    else
        fail "pybind11 C extension not built (falling back to pure Python)"
    fi

    if [[ $RUN_TESTS -eq 1 ]]; then
        # Test 1: native C extension (pybind11)
        run_step "Python tests (C backend)" $PY -m pytest tests/python/ -v

        # Test 2: force pure-Python fallback
        # Remove the compiled C extension so the fallback is used
        find "$ROOT/python" -name '_core*.so' -delete 2>/dev/null
        find "$ROOT/python" -name '_core*.pyd' -delete 2>/dev/null
        $PY -m pip install -e python/ --no-build-isolation --quiet 2>&1
        if $PY -c "import dress; assert dress._BACKEND == 'python'" 2>/dev/null; then
            run_step "Python tests (pure Python backend)" $PY -m pytest tests/python/ -v
        else
            fail "Could not switch to pure-Python backend"
        fi

        # Restore the native extension for subsequent use
        DRESS_BUILD_NATIVE=1 $PY -m pip install -e "python/[native]" --no-build-isolation --quiet 2>&1
    fi
}

# ── Rust ────────────────────────────────────────────────────────────
build_rust() {
    want rust || return 0
    if ! command -v cargo &>/dev/null; then
        skip "Rust"; return
    fi
    header "Rust"
    (cd rust && cargo build --release 2>&1)
    pass "Rust crate built"

    if [[ $RUN_TESTS -eq 1 ]]; then
        run_step "Rust tests" bash -c 'cd tests/rust && cargo test 2>&1'
    fi
}

# ── Go ──────────────────────────────────────────────────────────────
build_go() {
    want go || return 0
    if ! command -v go &>/dev/null; then
        skip "Go"; return
    fi
    header "Go"
    (cd go && CGO_ENABLED=1 go build ./... 2>&1)
    pass "Go package built"

    if [[ $RUN_TESTS -eq 1 ]]; then
        run_step "Go tests" bash -c 'cd tests/go && CGO_ENABLED=1 go test -v ./... 2>&1'
    fi
}

# ── R ───────────────────────────────────────────────────────────────
build_r() {
    want r || return 0
    if ! command -v R &>/dev/null; then
        skip "R"; return
    fi
    header "R"
    R CMD build r/ 2>&1 | tail -1
    pass "R tarball built"

    if [[ $RUN_TESTS -eq 1 ]]; then
        local tarball
        tarball=$(ls -t dress.graph_*.tar.gz 2>/dev/null | head -1)
        if [[ -n "$tarball" ]]; then
            run_step "R check" R CMD check "$tarball" --no-manual
        fi
    fi
}

# ── Julia ───────────────────────────────────────────────────────────
build_julia() {
    want julia || return 0
    if ! command -v julia &>/dev/null; then
        skip "Julia"; return
    fi
    header "Julia"
    if [[ $RUN_TESTS -eq 1 ]]; then
        run_step "Julia tests" julia tests/julia/test_dress.jl
    else
        # No separate build step — Julia is JIT compiled
        pass "Julia (JIT, no build step)"
    fi
}

# ── Octave (Forge tarball) ──────────────────────────────────────────
build_octave() {
    want octave || return 0
    header "Octave (Forge tarball)"

    # Detect version from DESCRIPTION
    local VERSION
    VERSION=$(grep -oP '^Version: \K.*' "$ROOT/octave/DESCRIPTION")
    if [[ -z "$VERSION" ]]; then
        fail "Octave (could not detect version)"; return
    fi

    # Assemble package in a temp directory
    local WORK PKG TARBALL
    WORK=$(mktemp -d)
    PKG="$WORK/dress-graph"
    mkdir -p "$PKG/inst/include/dress" "$PKG/src"

    # Octave scaffolding
    cp "$ROOT/octave/DESCRIPTION" "$PKG/"
    cp "$ROOT/octave/INDEX"       "$PKG/"
    cp "$ROOT/octave/PKG_ADD"     "$PKG/"
    cp "$ROOT/octave/PKG_DEL"     "$PKG/"
    cp "$ROOT/LICENSE"            "$PKG/COPYING" 2>/dev/null || true

    # Vendor .m files into inst/
    cp "$ROOT/matlab/dress_fit.m"        "$PKG/inst/"
    cp "$ROOT/matlab/delta_dress_fit.m"  "$PKG/inst/"
    cp "$ROOT/matlab/dress_to_table.m"   "$PKG/inst/"
    cp "$ROOT/matlab/DRESS.m"       "$PKG/inst/"

    # Vendor +cuda/ namespace (GPU wrappers)
    mkdir -p "$PKG/inst/+cuda"
    cp "$ROOT/matlab/+cuda/dress_fit.m"       "$PKG/inst/+cuda/"
    cp "$ROOT/matlab/+cuda/delta_dress_fit.m" "$PKG/inst/+cuda/"

    # Vendor C sources into src/
    cp "$ROOT/matlab/dress_mex.c"        "$PKG/src/"
    cp "$ROOT/matlab/delta_dress_mex.c"  "$PKG/src/"
    cp "$ROOT/matlab/dress_init_mex.c"   "$PKG/src/"
    cp "$ROOT/matlab/dress_fit_obj_mex.c" "$PKG/src/"
    cp "$ROOT/matlab/dress_get_mex.c"    "$PKG/src/"
    cp "$ROOT/matlab/dress_result_mex.c" "$PKG/src/"
    cp "$ROOT/matlab/dress_free_mex.c"   "$PKG/src/"
    cp "$ROOT/matlab/dress_cuda_mex.c"        "$PKG/src/"
    cp "$ROOT/matlab/delta_dress_cuda_mex.c"  "$PKG/src/"
    cp "$ROOT/libdress/src/cuda/delta_dress_cuda.c" "$PKG/src/"
    cp "$ROOT/libdress/src/cuda/dress_cuda.cu"      "$PKG/src/"
    cp "$ROOT/libdress/src/dress.c"          "$PKG/src/"
    cp "$ROOT/libdress/src/delta_dress.c"    "$PKG/src/"
    cp "$ROOT/libdress/src/delta_dress_impl.c" "$PKG/src/"
    cp "$ROOT/libdress/src/delta_dress_impl.h" "$PKG/src/"

    # Headers (used at build time via -I../inst/include)
    cp "$ROOT/libdress/include/dress/dress.h"       "$PKG/inst/include/dress/"
    cp "$ROOT/libdress/include/dress/delta_dress.h" "$PKG/inst/include/dress/"
    mkdir -p "$PKG/inst/include/dress/cuda"
    cp "$ROOT/libdress/include/dress/cuda/dress_cuda.h" "$PKG/inst/include/dress/cuda/"

    # Makefile
    cp "$ROOT/octave/src/Makefile" "$PKG/src/"

    # Build tarball
    TARBALL="$ROOT/dress-graph-${VERSION}.tar.gz"
    (cd "$WORK" && tar czf "$TARBALL" dress-graph/)

    rm -rf "$WORK"
    pass "Octave tarball: dress-graph-${VERSION}.tar.gz"
}

# ── WASM (Emscripten) ──────────────────────────────────────────────
build_wasm() {
    want wasm || return 0
    if ! command -v emcc &>/dev/null; then
        skip "WASM (emcc not found)"; return
    fi
    header "WASM"
    (cd wasm && bash build.sh 2>&1)
    pass "WASM built"

    if [[ $RUN_TESTS -eq 1 ]] && command -v node &>/dev/null; then
        run_step "WASM tests" bash -c 'cd tests/wasm && node test.mjs 2>&1'
        run_step "WASM delta tests" bash -c 'cd tests/wasm && node test_delta.mjs 2>&1'
    fi
}

# ── Run selected targets ───────────────────────────────────────────
vendor_sources
trap unvendor_sources EXIT

(want c || want cpp) && build_c_cpp
build_igraph
build_python
build_rust
build_go
build_r
build_julia
build_wasm
build_octave

# ── Summary ─────────────────────────────────────────────────────────
echo
echo "═══════════════════════════════════════════════"
echo "  Build complete:  $PASS passed,  $FAIL failed,  $SKIP skipped"
echo "═══════════════════════════════════════════════"

exit $FAIL
