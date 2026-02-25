#!/usr/bin/env bash
# Create (or refresh) the Python virtual environment and install dress-graph
# with all optional dependencies.
#
# Usage:
#   ./setup_env.sh          # create venv + install
#   source .venv/bin/activate   # activate (run manually afterwards)

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"

# --- create venv if missing ---
if [[ ! -d "$VENV" ]]; then
    echo "Creating virtual environment at $VENV ..."
    python3 -m venv "$VENV"
fi

# --- activate ---
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- upgrade pip ---
pip install --upgrade pip setuptools wheel -q

# --- install dress-graph in editable mode with all extras ---
pip install -e "$ROOT/python[native,networkx]" -q

# --- native build dependencies (C extension via pybind11) ---
pip install pybind11 numpy -q

# --- dev / test dependencies ---
pip install pytest mkdocs pymdown-extensions build twine -q

# --- Emscripten SDK (WASM) ---
if [[ -f "$HOME/emsdk/emsdk_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/emsdk/emsdk_env.sh" 2>/dev/null
    echo "Emscripten SDK activated ($(emcc --version | head -1))"
else
    echo "Note: emsdk not found at ~/emsdk — WASM target will be skipped."
fi

echo ""
echo "Environment ready.  Activate with:"
echo "  source $VENV/bin/activate"
