#!/usr/bin/env bash
# build.sh — Build and test every language binding in the dress-graph repo.
#
# Usage:
#   ./build.sh            # build + test everything available
#   ./build.sh --no-test  # build only, skip tests
#   ./build.sh c cpp      # build + test only the listed targets
#
# Targets: c cpp python rust go r julia wasm
#
# A target is silently skipped when its toolchain is not installed.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

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
    TARGETS=(c cpp python rust go r julia wasm)
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
    mkdir -p rust/vendor/include/dress
    cp libdress/src/dress.c        rust/vendor/dress.c
    cp libdress/include/dress/dress.h rust/vendor/include/dress/dress.h

    # R
    mkdir -p r/src/dress
    cp libdress/src/dress.c        r/src/dress.c
    cp libdress/include/dress/dress.h r/src/dress/dress.h
}

# Remove vendored copies.
unvendor_sources() {
    rm -rf rust/vendor
    rm -f  r/src/dress.c
    rm -rf r/src/dress
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
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1
    cmake --build build -j"$(nproc 2>/dev/null || echo 2)" 2>&1
    pass "libdress built"

    if [[ $RUN_TESTS -eq 1 ]]; then
        # C tests
        if want c && [[ -f tests/c/test_dress.c ]]; then
            gcc -O2 -o tests/c/test_dress tests/c/test_dress.c \
                -Ilibdress/include -Lbuild/libdress -ldress -lm -fopenmp 2>&1
            run_step "C tests" env LD_LIBRARY_PATH=build/libdress tests/c/test_dress
        fi

        # C++ tests
        if want cpp && [[ -f tests/cpp/test_dress.cpp ]]; then
            g++ -O2 -std=c++17 -o tests/cpp/test_dress tests/cpp/test_dress.cpp \
                -Ilibdress/include -Ilibdress++/include \
                -Lbuild/libdress -ldress -lm -fopenmp 2>&1
            run_step "C++ tests" env LD_LIBRARY_PATH=build/libdress tests/cpp/test_dress
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
        run_step "WASM tests" bash -c 'cd wasm && node test.mjs 2>&1'
    fi
}

# ── Run selected targets ───────────────────────────────────────────
vendor_sources
trap unvendor_sources EXIT

(want c || want cpp) && build_c_cpp
build_python
build_rust
build_go
build_r
build_julia
build_wasm

# ── Summary ─────────────────────────────────────────────────────────
echo
echo "═══════════════════════════════════════════════"
echo "  Build complete:  $PASS passed,  $FAIL failed,  $SKIP skipped"
echo "═══════════════════════════════════════════════"

exit $FAIL
