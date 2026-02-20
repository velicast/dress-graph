#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  run_tests.sh — Run all DRESS test suites.
#
#  Usage:
#    ./run_tests.sh            # run all suites (skip unavailable ones)
#    ./run_tests.sh c cpp      # run only C and C++ suites
#
#  Exit code: 0 if all requested suites pass, 1 otherwise.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PASS=0
FAIL=0
SKIP=0
RESULTS=()

# ── colours (disabled when not a tty) ────────────────────────────────
if [ -t 1 ]; then
    GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'
    BOLD='\033[1m'; RESET='\033[0m'
else
    GREEN=''; RED=''; YELLOW=''; BOLD=''; RESET=''
fi

pass()  { PASS=$((PASS+1)); RESULTS+=("${GREEN}  PASS${RESET}  $1"); }
fail()  { FAIL=$((FAIL+1)); RESULTS+=("${RED}  FAIL${RESET}  $1"); }
skip()  { SKIP=$((SKIP+1)); RESULTS+=("${YELLOW}  SKIP${RESET}  $1"); }

# ── suite: C ─────────────────────────────────────────────────────────
run_c() {
    echo -e "\n${BOLD}── C ──${RESET}"
    if ! command -v gcc &>/dev/null; then skip "c (gcc not found)"; return; fi
    gcc -O2 -I "$ROOT/libdress/include" \
        -o "$ROOT/tests/c/test_dress" \
        "$ROOT/tests/c/test_dress.c" "$ROOT/libdress/src/dress.c" \
        -lm -fopenmp \
    && "$ROOT/tests/c/test_dress" \
    && pass "c" || fail "c"
}

# ── suite: C++ ───────────────────────────────────────────────────────
run_cpp() {
    echo -e "\n${BOLD}── C++ ──${RESET}"
    if ! command -v g++ &>/dev/null; then skip "cpp (g++ not found)"; return; fi
    g++ -std=c++17 -O2 \
        -I "$ROOT/libdress/include" -I "$ROOT/libdress++/include" \
        -o "$ROOT/tests/cpp/test_dress" \
        "$ROOT/tests/cpp/test_dress.cpp" "$ROOT/libdress/src/dress.c" \
        -lm -fopenmp \
    && "$ROOT/tests/cpp/test_dress" \
    && pass "cpp" || fail "cpp"
}

# ── suite: Rust ──────────────────────────────────────────────────────
run_rust() {
    echo -e "\n${BOLD}── Rust ──${RESET}"
    if ! command -v cargo &>/dev/null; then skip "rust (cargo not found)"; return; fi
    (cd "$ROOT/tests/rust" && cargo test 2>&1) \
    && pass "rust" || fail "rust"
}

# ── suite: Go ────────────────────────────────────────────────────────
run_go() {
    echo -e "\n${BOLD}── Go ──${RESET}"
    if ! command -v go &>/dev/null; then skip "go (go not found)"; return; fi
    (cd "$ROOT/tests/go" && go test -v ./... 2>&1) \
    && pass "go" || fail "go"
}

# ── suite: Python ────────────────────────────────────────────────────
run_python() {
    echo -e "\n${BOLD}── Python ──${RESET}"
    local PYTHON=""
    # Prefer the venv if it exists
    if [ -x "$ROOT/.venv/bin/python" ]; then
        PYTHON="$ROOT/.venv/bin/python"
    elif command -v python3 &>/dev/null; then
        PYTHON="python3"
    else
        skip "python (python3 not found)"; return
    fi
    if ! "$PYTHON" -c "import pytest" &>/dev/null; then
        skip "python (pytest not installed)"; return
    fi
    if ! "$PYTHON" -c "import dress" &>/dev/null; then
        skip "python (dress extension not built — run: pip install ./python)"; return
    fi
    "$PYTHON" -m pytest "$ROOT/tests/python" -v 2>&1 \
    && pass "python" || fail "python"
}

# ── suite: WASM ──────────────────────────────────────────────────────
run_wasm() {
    echo -e "\n${BOLD}── WASM ──${RESET}"
    if ! command -v node &>/dev/null; then skip "wasm (node not found)"; return; fi
    if [ ! -f "$ROOT/wasm/dress.js" ]; then
        skip "wasm (not built — run: cd wasm && bash build.sh)"; return
    fi
    (cd "$ROOT/tests/wasm" && node test.mjs 2>&1) \
    && pass "wasm" || fail "wasm"
}

# ── suite: Julia ─────────────────────────────────────────────────────
run_julia() {
    echo -e "\n${BOLD}── Julia ──${RESET}"
    if ! command -v julia &>/dev/null; then skip "julia (julia not found)"; return; fi
    julia "$ROOT/tests/julia/test_dress.jl" 2>&1 \
    && pass "julia" || fail "julia"
}

# ── suite: C (igraph) ────────────────────────────────────────────────
run_igraph() {
    echo -e "\n${BOLD}── C (igraph) ──${RESET}"
    if ! command -v gcc &>/dev/null; then skip "igraph (gcc not found)"; return; fi
    if ! pkg-config --exists igraph 2>/dev/null; then
        skip "igraph (libigraph not installed)"; return
    fi
    gcc -O2 -I "$ROOT/libdress/include" -I "$ROOT/libdress-igraph/include" \
        $(pkg-config --cflags igraph) \
        -o "$ROOT/tests/c/test_dress_igraph" \
        "$ROOT/tests/c/test_dress_igraph.c" \
        "$ROOT/libdress-igraph/src/dress_igraph.c" \
        "$ROOT/libdress/src/dress.c" \
        $(pkg-config --libs igraph) -lm -fopenmp \
    && "$ROOT/tests/c/test_dress_igraph" \
    && pass "igraph" || fail "igraph"
}

# ── dispatch ─────────────────────────────────────────────────────────
ALL_SUITES=(c cpp rust go python julia wasm igraph)

if [ $# -gt 0 ]; then
    SUITES=("$@")
else
    SUITES=("${ALL_SUITES[@]}")
fi

for suite in "${SUITES[@]}"; do
    case "$suite" in
        c)      run_c      ;;
        cpp)    run_cpp    ;;
        rust)   run_rust   ;;
        go)     run_go     ;;
        python) run_python ;;
        julia)  run_julia  ;;
        wasm)   run_wasm   ;;
        igraph) run_igraph ;;
        *)      echo "Unknown suite: $suite"; fail "$suite" ;;
    esac
done

# ── summary ──────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════${RESET}"
echo -e "${BOLD}  Test Summary${RESET}"
echo -e "${BOLD}════════════════════════════════════${RESET}"
for r in "${RESULTS[@]}"; do echo -e "$r"; done
echo -e "${BOLD}────────────────────────────────────${RESET}"
echo -e "  ${GREEN}$PASS passed${RESET}, ${RED}$FAIL failed${RESET}, ${YELLOW}$SKIP skipped${RESET}"
echo -e "${BOLD}════════════════════════════════════${RESET}"

exit $((FAIL > 0 ? 1 : 0))
