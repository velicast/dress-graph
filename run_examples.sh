#!/usr/bin/env bash
# run_examples.sh — Install local packages and run all DRESS examples.
#
# Usage:
#   ./run_examples.sh [LANG...]
#
# Languages: c cpp python rust go julia r octave wasm all (default: all)
#
# For each language the script:
#   1. Installs the local package (publish.sh --install-local or direct build)
#   2. Compiles & runs: cpu, cuda, mpi, mpi_cuda examples
#   3. Verifies output correctness
#
# CUDA examples are skipped when nvcc is not found.
# MPI examples are skipped when mpicc/mpirun are not found.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Parse args ──────────────────────────────────────────────────────
LANGS=()
for arg in "$@"; do
    LANGS+=("$arg")
done
if [[ ${#LANGS[@]} -eq 0 ]]; then
    LANGS=(all)
fi

want() {
    [[ " ${LANGS[*]} " == *" all "* ]] || [[ " ${LANGS[*]} " == *" $1 "* ]]
}

# ── Feature detection ──────────────────────────────────────────────
HAS_CUDA=0
HAS_MPI=0
command -v nvcc   &>/dev/null && HAS_CUDA=1
command -v mpicc  &>/dev/null && command -v mpirun &>/dev/null && HAS_MPI=1

# ── Counters ────────────────────────────────────────────────────────
PASS=0
FAIL=0
SKIP=0

pass()  { echo "  ✓ $1"; PASS=$((PASS + 1)); }
fail()  { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }
skip()  { echo "  – $1 (skipped)"; SKIP=$((SKIP + 1)); }
header(){ echo; echo "══ $1 ══════════════════════════════════════════"; }

# ── Verification helpers ────────────────────────────────────────────
# CPU/CUDA examples: output must contain "Distinguished" with a truthy value
verify_cpu() {
    local label="$1" output="$2"
    if echo "$output" | grep -qiE 'Distinguished.*([Tt]rue|[Yy]es|fingerprints differ)'; then
        pass "$label"
    else
        fail "$label"
        echo "    OUTPUT: $output"
    fi
}

# MPI/MPI+CUDA examples: output must contain both histograms and multisets differing
verify_mpi() {
    local label="$1" output="$2"
    local ok=1
    if ! echo "$output" | grep -qiE 'Histograms differ.*([Tt]rue|[Yy]es)'; then
        ok=0
    fi
    if ! echo "$output" | grep -qiE 'Multisets differ.*([Tt]rue|[Yy]es)'; then
        ok=0
    fi
    if [[ $ok -eq 1 ]]; then
        pass "$label"
    else
        fail "$label"
        echo "    OUTPUT: $output"
    fi
}

# MPI igraph examples: only histograms (no multiset output)
verify_mpi_igraph() {
    local label="$1" output="$2"
    if echo "$output" | grep -qiE 'Histograms differ.*([Tt]rue|[Yy]es)'; then
        pass "$label"
    else
        fail "$label"
        echo "    OUTPUT: $output"
    fi
}

# ── Library paths for C/C++ examples ───────────────────────────────
LIBDIR="$ROOT/build/libdress"
INC_C="$ROOT/libdress/include"
INC_CPP="$ROOT/libdress++/include"
INC_IGRAPH="$ROOT/libdress-igraph/include"
export LD_LIBRARY_PATH="${LIBDIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_LIBS=""
if [[ -f "$LIBDIR/libdress_cuda.so" ]]; then
    CUDA_LIBS="-ldress_cuda -lcudart"
fi

# ═══════════════════════════════════════════════════════════════════
# C
# ═══════════════════════════════════════════════════════════════════
run_c() {
    header "C"
    local dir="$ROOT/examples/c"
    local out

    # cpu
    if gcc -O2 -o "$dir/cpu" "$dir/cpu.c" -I"$INC_C" -L"$LIBDIR" -ldress -lm 2>&1; then
        out=$("$dir/cpu" 2>&1) || true
        verify_cpu "C cpu" "$out"
    else
        fail "C cpu (compile)"
    fi

    # cuda
    if [[ $HAS_CUDA -eq 1 && -n "$CUDA_LIBS" ]]; then
        if gcc -O2 -o "$dir/cuda" "$dir/cuda.c" -I"$INC_C" -L"$LIBDIR" -ldress $CUDA_LIBS -lm 2>&1; then
            out=$("$dir/cuda" 2>&1) || true
            verify_cpu "C cuda" "$out"
        else
            fail "C cuda (compile)"
        fi
    else
        skip "C cuda (no CUDA)"
    fi

    # mpi
    if [[ $HAS_MPI -eq 1 ]]; then
        if mpicc -O2 -o "$dir/mpi" "$dir/mpi.c" -I"$INC_C" -L"$LIBDIR" -ldress -lm 2>&1; then
            out=$(mpirun --oversubscribe -np 4 "$dir/mpi" 2>&1) || true
            verify_mpi "C mpi" "$out"
        else
            fail "C mpi (compile)"
        fi
    else
        skip "C mpi (no MPI)"
    fi

    # mpi_cuda
    if [[ $HAS_MPI -eq 1 && $HAS_CUDA -eq 1 && -n "$CUDA_LIBS" ]]; then
        if mpicc -O2 -o "$dir/mpi_cuda" "$dir/mpi_cuda.c" -I"$INC_C" -L"$LIBDIR" -ldress $CUDA_LIBS -lm 2>&1; then
            out=$(mpirun --oversubscribe -np 4 "$dir/mpi_cuda" 2>&1) || true
            verify_mpi "C mpi_cuda" "$out"
        else
            fail "C mpi_cuda (compile)"
        fi
    else
        skip "C mpi_cuda (no MPI+CUDA)"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# C igraph
# ═══════════════════════════════════════════════════════════════════
run_c_igraph() {
    if ! pkg-config --exists igraph 2>/dev/null; then
        skip "C igraph (pkg-config igraph not found)"; return
    fi
    header "C igraph"
    local dir="$ROOT/examples/c"
    local IGRAPH_CFLAGS IGRAPH_LIBS
    IGRAPH_CFLAGS=$(pkg-config --cflags igraph)
    IGRAPH_LIBS=$(pkg-config --libs igraph)
    local SRC_IGRAPH="$ROOT/libdress-igraph/src/dress_igraph.c"
    local out

    # cpu_igraph
    if gcc -O2 -o "$dir/cpu_igraph" "$dir/cpu_igraph.c" $SRC_IGRAPH \
            -I"$INC_C" -I"$INC_IGRAPH" -I"$ROOT/libdress/src" \
            $IGRAPH_CFLAGS -L"$LIBDIR" -ldress $IGRAPH_LIBS -lm -fopenmp 2>&1; then
        out=$("$dir/cpu_igraph" 2>&1) || true
        verify_cpu "C igraph cpu" "$out"
    else
        fail "C igraph cpu (compile)"
    fi

    # cuda_igraph
    if [[ $HAS_CUDA -eq 1 && -n "$CUDA_LIBS" ]]; then
        if gcc -O2 -o "$dir/cuda_igraph" "$dir/cuda_igraph.c" $SRC_IGRAPH \
                -I"$INC_C" -I"$INC_IGRAPH" -I"$ROOT/libdress/src" \
                $IGRAPH_CFLAGS -L"$LIBDIR" -ldress $CUDA_LIBS $IGRAPH_LIBS -lm -fopenmp 2>&1; then
            out=$("$dir/cuda_igraph" 2>&1) || true
            verify_cpu "C igraph cuda" "$out"
        else
            fail "C igraph cuda (compile)"
        fi
    else
        skip "C igraph cuda (no CUDA)"
    fi

    # mpi_igraph
    if [[ $HAS_MPI -eq 1 ]]; then
        local SRC_MPI_IGRAPH="$ROOT/libdress-igraph/src/dress_igraph_mpi.c"
        if mpicc -O2 -o "$dir/mpi_igraph" "$dir/mpi_igraph.c" \
                $SRC_IGRAPH $SRC_MPI_IGRAPH \
                -I"$INC_C" -I"$INC_IGRAPH" -I"$ROOT/libdress/src" \
                $IGRAPH_CFLAGS -L"$LIBDIR" -ldress $IGRAPH_LIBS -lm -fopenmp 2>&1; then
            out=$(mpirun --oversubscribe -np 4 "$dir/mpi_igraph" 2>&1) || true
            verify_mpi_igraph "C igraph mpi" "$out"
        else
            fail "C igraph mpi (compile)"
        fi
    else
        skip "C igraph mpi (no MPI)"
    fi

    # mpi_cuda_igraph
    if [[ $HAS_MPI -eq 1 && $HAS_CUDA -eq 1 && -n "$CUDA_LIBS" ]]; then
        local SRC_MPI_IGRAPH="$ROOT/libdress-igraph/src/dress_igraph_mpi.c"
        if mpicc -O2 -DDRESS_CUDA -o "$dir/mpi_cuda_igraph" "$dir/mpi_cuda_igraph.c" \
                $SRC_IGRAPH $SRC_MPI_IGRAPH \
                -I"$INC_C" -I"$INC_IGRAPH" -I"$ROOT/libdress/src" \
                $IGRAPH_CFLAGS -L"$LIBDIR" -ldress $CUDA_LIBS $IGRAPH_LIBS -lm -fopenmp 2>&1; then
            out=$(mpirun --oversubscribe -np 4 "$dir/mpi_cuda_igraph" 2>&1) || true
            verify_mpi_igraph "C igraph mpi_cuda" "$out"
        else
            fail "C igraph mpi_cuda (compile)"
        fi
    else
        skip "C igraph mpi_cuda (no MPI+CUDA)"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# C++
# ═══════════════════════════════════════════════════════════════════
run_cpp() {
    header "C++"
    local dir="$ROOT/examples/cpp"
    local out

    # cpu
    if g++ -O2 -std=c++17 -o "$dir/cpu" "$dir/cpu.cpp" \
            -I"$INC_C" -I"$INC_CPP" -L"$LIBDIR" -ldress -lm 2>&1; then
        out=$("$dir/cpu" 2>&1) || true
        verify_cpu "C++ cpu" "$out"
    else
        fail "C++ cpu (compile)"
    fi

    # cuda
    if [[ $HAS_CUDA -eq 1 && -n "$CUDA_LIBS" ]]; then
        if g++ -O2 -std=c++17 -o "$dir/cuda" "$dir/cuda.cpp" \
                -I"$INC_C" -I"$INC_CPP" -L"$LIBDIR" -ldress $CUDA_LIBS -lm 2>&1; then
            out=$("$dir/cuda" 2>&1) || true
            verify_cpu "C++ cuda" "$out"
        else
            fail "C++ cuda (compile)"
        fi
    else
        skip "C++ cuda (no CUDA)"
    fi

    # mpi
    if [[ $HAS_MPI -eq 1 ]]; then
        if mpicxx -O2 -std=c++17 -o "$dir/mpi" "$dir/mpi.cpp" \
                -I"$INC_C" -I"$INC_CPP" -L"$LIBDIR" -ldress -lm 2>&1; then
            out=$(mpirun --oversubscribe -np 4 "$dir/mpi" 2>&1) || true
            verify_mpi "C++ mpi" "$out"
        else
            fail "C++ mpi (compile)"
        fi
    else
        skip "C++ mpi (no MPI)"
    fi

    # mpi_cuda
    if [[ $HAS_MPI -eq 1 && $HAS_CUDA -eq 1 && -n "$CUDA_LIBS" ]]; then
        if mpicxx -O2 -std=c++17 -o "$dir/mpi_cuda" "$dir/mpi_cuda.cpp" \
                -I"$INC_C" -I"$INC_CPP" -L"$LIBDIR" -ldress $CUDA_LIBS -lm 2>&1; then
            out=$(mpirun --oversubscribe -np 4 "$dir/mpi_cuda" 2>&1) || true
            verify_mpi "C++ mpi_cuda" "$out"
        else
            fail "C++ mpi_cuda (compile)"
        fi
    else
        skip "C++ mpi_cuda (no MPI+CUDA)"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# Python
# ═══════════════════════════════════════════════════════════════════
run_python() {
    header "Python"

    local PY="python3"
    if [[ -f "$ROOT/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$ROOT/.venv/bin/activate"
        PY=python
    fi

    # Install local package
    echo "  Installing Python package locally ..."
    bash "$ROOT/publish.sh" --install-local pypi 2>&1 | tail -3

    # Ensure mpi4py + numpy + networkx are available
    $PY -m pip install networkx numpy --quiet 2>&1 || true
    if [[ $HAS_MPI -eq 1 ]]; then
        $PY -m pip install mpi4py --quiet 2>&1 || true
    fi
    echo

    local dir="$ROOT/examples/python"
    local out

    # cpu
    out=$($PY "$dir/cpu.py" 2>&1) || true
    verify_cpu "Python cpu" "$out"

    # cpu_nx
    out=$($PY "$dir/cpu_nx.py" 2>&1) || true
    verify_cpu "Python cpu_nx" "$out"

    # cuda
    if [[ $HAS_CUDA -eq 1 ]]; then
        out=$($PY "$dir/cuda.py" 2>&1) || true
        verify_cpu "Python cuda" "$out"
    else
        skip "Python cuda (no CUDA)"
    fi

    # cuda_nx
    if [[ $HAS_CUDA -eq 1 ]]; then
        out=$($PY "$dir/cuda_nx.py" 2>&1) || true
        verify_cpu "Python cuda_nx" "$out"
    else
        skip "Python cuda_nx (no CUDA)"
    fi

    # mpi
    if [[ $HAS_MPI -eq 1 ]]; then
        out=$(mpirun --oversubscribe -np 4 $PY "$dir/mpi.py" 2>&1) || true
        verify_mpi "Python mpi" "$out"
    else
        skip "Python mpi (no MPI)"
    fi

    # mpi_nx
    if [[ $HAS_MPI -eq 1 ]]; then
        out=$(mpirun --oversubscribe -np 4 $PY "$dir/mpi_nx.py" 2>&1) || true
        verify_mpi "Python mpi_nx" "$out"
    else
        skip "Python mpi_nx (no MPI)"
    fi

    # mpi_cuda
    if [[ $HAS_MPI -eq 1 && $HAS_CUDA -eq 1 ]]; then
        out=$(mpirun --oversubscribe -np 4 $PY "$dir/mpi_cuda.py" 2>&1) || true
        verify_mpi "Python mpi_cuda" "$out"
    else
        skip "Python mpi_cuda (no MPI+CUDA)"
    fi

    # mpi_cuda_nx
    if [[ $HAS_MPI -eq 1 && $HAS_CUDA -eq 1 ]]; then
        out=$(mpirun --oversubscribe -np 4 $PY "$dir/mpi_cuda_nx.py" 2>&1) || true
        verify_mpi "Python mpi_cuda_nx" "$out"
    else
        skip "Python mpi_cuda_nx (no MPI+CUDA)"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# Rust
# ═══════════════════════════════════════════════════════════════════
run_rust() {
    header "Rust"
    if ! command -v cargo &>/dev/null; then
        skip "Rust (cargo not found)"; return
    fi

    # Install (vendor sources + build)
    echo "  Installing Rust package locally ..."
    bash "$ROOT/publish.sh" --install-local cargo 2>&1 | tail -3
    echo

    local dir="$ROOT/examples/rust"
    local out

    # cpu
    out=$(cd "$dir" && cargo run --example cpu 2>&1) || true
    verify_cpu "Rust cpu" "$out"

    # cuda
    if [[ $HAS_CUDA -eq 1 ]]; then
        out=$(cd "$dir" && cargo run --example cuda --features cuda 2>&1) || true
        verify_cpu "Rust cuda" "$out"
    else
        skip "Rust cuda (no CUDA)"
    fi

    # mpi
    if [[ $HAS_MPI -eq 1 ]]; then
        (cd "$dir" && cargo build --example mpi --features mpi 2>&1) || true
        local bin
        bin=$(find "$dir/target" -name mpi -type f -executable 2>/dev/null | head -1)
        if [[ -n "$bin" ]]; then
            out=$(mpirun --oversubscribe -np 4 "$bin" 2>&1) || true
            verify_mpi "Rust mpi" "$out"
        else
            fail "Rust mpi (binary not found)"
        fi
    else
        skip "Rust mpi (no MPI)"
    fi

    # mpi_cuda
    if [[ $HAS_MPI -eq 1 && $HAS_CUDA -eq 1 ]]; then
        (cd "$dir" && cargo build --example mpi_cuda --features "mpi cuda" 2>&1) || true
        local bin
        bin=$(find "$dir/target" -name mpi_cuda -type f -executable 2>/dev/null | head -1)
        if [[ -n "$bin" ]]; then
            out=$(mpirun --oversubscribe -np 4 "$bin" 2>&1) || true
            verify_mpi "Rust mpi_cuda" "$out"
        else
            fail "Rust mpi_cuda (binary not found)"
        fi
    else
        skip "Rust mpi_cuda (no MPI+CUDA)"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# Go
# ═══════════════════════════════════════════════════════════════════
run_go() {
    header "Go"
    if ! command -v go &>/dev/null; then
        skip "Go (go not found)"; return
    fi

    local dir="$ROOT/examples/go"
    local out

    # cpu
    out=$(cd "$dir" && CGO_ENABLED=1 \
        CGO_CFLAGS="-I$INC_C" \
        CGO_LDFLAGS="-L$LIBDIR -ldress -lm" \
        go run cpu.go 2>&1) || true
    verify_cpu "Go cpu" "$out"

    # cuda
    if [[ $HAS_CUDA -eq 1 && -n "$CUDA_LIBS" ]]; then
        out=$(cd "$dir" && CGO_ENABLED=1 \
            CGO_CFLAGS="-I$INC_C" \
            CGO_LDFLAGS="-L$LIBDIR -ldress -ldress_cuda -lcudart -lm" \
            go run cuda.go 2>&1) || true
        verify_cpu "Go cuda" "$out"
    else
        skip "Go cuda (no CUDA)"
    fi

    # mpi
    if [[ $HAS_MPI -eq 1 ]]; then
        (cd "$dir" && CGO_ENABLED=1 \
            CGO_CFLAGS="-I$INC_C" \
            CGO_LDFLAGS="-L$LIBDIR -ldress -lm" \
            go build -o mpi_bin mpi.go 2>&1) || true
        if [[ -f "$dir/mpi_bin" ]]; then
            out=$(mpirun --oversubscribe -np 4 "$dir/mpi_bin" 2>&1) || true
            verify_mpi "Go mpi" "$out"
            rm -f "$dir/mpi_bin"
        else
            fail "Go mpi (build)"
        fi
    else
        skip "Go mpi (no MPI)"
    fi

    # mpi_cuda
    if [[ $HAS_MPI -eq 1 && $HAS_CUDA -eq 1 && -n "$CUDA_LIBS" ]]; then
        (cd "$dir" && CGO_ENABLED=1 \
            CGO_CFLAGS="-I$INC_C" \
            CGO_LDFLAGS="-L$LIBDIR -ldress -ldress_cuda -lcudart -lm" \
            go build -o mpi_cuda_bin mpi_cuda.go 2>&1) || true
        if [[ -f "$dir/mpi_cuda_bin" ]]; then
            out=$(mpirun --oversubscribe -np 4 "$dir/mpi_cuda_bin" 2>&1) || true
            verify_mpi "Go mpi_cuda" "$out"
            rm -f "$dir/mpi_cuda_bin"
        else
            fail "Go mpi_cuda (build)"
        fi
    else
        skip "Go mpi_cuda (no MPI+CUDA)"
    fi

}

# ═══════════════════════════════════════════════════════════════════
# Julia
# ═══════════════════════════════════════════════════════════════════
run_julia() {
    header "Julia"
    if ! command -v julia &>/dev/null; then
        skip "Julia (julia not found)"; return
    fi

    # Install local package
    echo "  Installing Julia package locally ..."
    bash "$ROOT/publish.sh" --install-local julia 2>&1 | tail -3
    echo

    local dir="$ROOT/examples/julia"
    local out

    # cpu
    out=$(julia "$dir/cpu.jl" 2>&1) || true
    verify_cpu "Julia cpu" "$out"

    # cuda
    if [[ $HAS_CUDA -eq 1 ]]; then
        out=$(julia "$dir/cuda.jl" 2>&1) || true
        verify_cpu "Julia cuda" "$out"
    else
        skip "Julia cuda (no CUDA)"
    fi

    # mpi
    if [[ $HAS_MPI -eq 1 ]]; then
        out=$(mpirun --oversubscribe -np 4 julia "$dir/mpi.jl" 2>&1) || true
        verify_mpi "Julia mpi" "$out"
    else
        skip "Julia mpi (no MPI)"
    fi

    # mpi_cuda
    if [[ $HAS_MPI -eq 1 && $HAS_CUDA -eq 1 ]]; then
        out=$(mpirun --oversubscribe -np 4 julia "$dir/mpi_cuda.jl" 2>&1) || true
        verify_mpi "Julia mpi_cuda" "$out"
    else
        skip "Julia mpi_cuda (no MPI+CUDA)"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# R
# ═══════════════════════════════════════════════════════════════════
run_r() {
    header "R"
    if ! command -v R &>/dev/null || ! command -v Rscript &>/dev/null; then
        skip "R (R/Rscript not found)"; return
    fi

    # Install local package
    echo "  Installing R package locally ..."
    bash "$ROOT/publish.sh" --install-local cran 2>&1 | tail -3
    echo

    local dir="$ROOT/examples/r"
    local out

    # cpu
    out=$(Rscript "$dir/cpu.R" 2>&1) || true
    verify_cpu "R cpu" "$out"

    # cuda
    if [[ $HAS_CUDA -eq 1 ]]; then
        out=$(Rscript "$dir/cuda.R" 2>&1) || true
        verify_cpu "R cuda" "$out"
    else
        skip "R cuda (no CUDA)"
    fi

    # mpi
    if [[ $HAS_MPI -eq 1 ]]; then
        out=$(mpirun --oversubscribe -np 4 Rscript "$dir/mpi.R" 2>&1) || true
        verify_mpi "R mpi" "$out"
    else
        skip "R mpi (no MPI)"
    fi

    # mpi_cuda
    if [[ $HAS_MPI -eq 1 && $HAS_CUDA -eq 1 ]]; then
        out=$(mpirun --oversubscribe -np 4 Rscript "$dir/mpi_cuda.R" 2>&1) || true
        verify_mpi "R mpi_cuda" "$out"
    else
        skip "R mpi_cuda (no MPI+CUDA)"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# Octave
# ═══════════════════════════════════════════════════════════════════
run_octave() {
    header "Octave"

    if ! command -v octave &>/dev/null; then
        skip "Octave (octave not found)"
        return
    fi

    # Install octave package locally
    local tarball="$ROOT/dress-graph-0.5.0.tar.gz"
    if [[ ! -f "$tarball" ]]; then
        echo "  Building Octave tarball ..."
        bash "$ROOT/build.sh" octave 2>&1 | tail -1
    fi
    echo "  Installing Octave package locally ..."
    octave --eval "pkg install '$tarball'" 2>&1 | tail -3
    echo "  ✓ installed locally"

    local dir="$ROOT/examples/octave"
    local out

    # cpu
    out=$(octave --no-gui --eval "
        pkg load dress-graph;
        run('$dir/cpu.m');
    " 2>&1) || true
    verify_cpu "Octave cpu" "$out"

    # cuda
    if [[ $HAS_CUDA -eq 1 ]]; then
        out=$(octave --no-gui --eval "
            pkg load dress-graph;
            run('$dir/cuda_example.m');
        " 2>&1) || true
        verify_cpu "Octave cuda" "$out"
    else
        skip "Octave cuda (no CUDA)"
    fi

    # rook vs shrikhande (delta-1)
    out=$(octave --no-gui --eval "
        pkg load dress-graph;
        run('$dir/rook_vs_shrikhande.m');
    " 2>&1) || true
    verify_mpi "Octave rook_vs_shrikhande" "$out"
}

# ═══════════════════════════════════════════════════════════════════
# WASM (Node.js)
# ═══════════════════════════════════════════════════════════════════
run_wasm() {
    header "WASM"

    if ! command -v node &>/dev/null; then
        skip "WASM (node not found)"
        return
    fi

    if [[ ! -f "$ROOT/wasm/dress_wasm.wasm" ]]; then
        skip "WASM (dress_wasm.wasm not built — run build.sh wasm first)"
        return
    fi

    local dir="$ROOT/examples/wasm"
    local out

    # cpu
    out=$(node "$dir/cpu.mjs" 2>&1) || true
    verify_cpu "WASM cpu" "$out"

    # rook vs shrikhande (delta-1)
    out=$(node "$dir/rook_vs_shrikhande.mjs" 2>&1) || true
    verify_mpi "WASM rook_vs_shrikhande" "$out"
}

# ═══════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════
echo "DRESS Examples Runner"
echo "  CUDA: $([ $HAS_CUDA -eq 1 ] && echo 'available' || echo 'not found')"
echo "  MPI:  $([ $HAS_MPI  -eq 1 ] && echo 'available' || echo 'not found')"

want c      && run_c
want igraph && run_c_igraph
want cpp    && run_cpp
want python && run_python
want rust   && run_rust
want go     && run_go
want julia  && run_julia
want r      && run_r
want octave && run_octave
want wasm   && run_wasm

# ── Summary ─────────────────────────────────────────────────────────
echo
echo "═══════════════════════════════════════════════"
echo "  Examples:  $PASS passed,  $FAIL failed,  $SKIP skipped"
echo "═══════════════════════════════════════════════"

exit $FAIL
