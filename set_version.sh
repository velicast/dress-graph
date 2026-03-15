#!/usr/bin/env bash
# set_version.sh — Update the package version across all language bindings.
#
# Usage:
#   ./set_version.sh 0.1.2
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.1.2"
    exit 1
fi

NEW="$1"

# Validate semver-ish format (X.Y.Z with optional pre-release)
if ! [[ "$NEW" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    echo "ERROR: '$NEW' is not a valid version (expected X.Y.Z)"
    exit 1
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CHANGED=0

# Helper: sed in-place, portable (GNU + macOS)
sedi() {
    sed -i "$@"
}

update() {
    local file="$1" pattern="$2" replacement="$3"
    if [[ -f "$file" ]]; then
        if grep -qE "$pattern" "$file"; then
            sedi -E "s|$pattern|$replacement|" "$file"
            echo "  ✓ $file"
            CHANGED=$((CHANGED + 1))
        fi
    fi
}

echo "Setting version to $NEW ..."
echo

# Python — pyproject.toml
update python/pyproject.toml \
    '^version = ".*"' \
    "version = \"$NEW\""

# Rust — Cargo.toml
update rust/Cargo.toml \
    '^version *= *".*"' \
    "version = \"$NEW\""

# npm/WASM — package.json
update wasm/package.json \
    '"version": ".*"' \
    "\"version\": \"$NEW\""

# R — DESCRIPTION
update r/DESCRIPTION \
    '^Version: .*' \
    "Version: $NEW"

# R — dress_r.c (runtime version string)
update r/src/dress_r.c \
    'return ScalarString\(mkChar\(".*"\)\);' \
    "return ScalarString(mkChar(\"$NEW\"));"

# Julia — Project.toml
update julia/Project.toml \
    '^version = ".*"' \
    "version = \"$NEW\""

# Octave — DESCRIPTION
update octave/DESCRIPTION \
    '^Version: .*' \
    "Version: $NEW"

# Docs — Octave install URLs in installation.md
if [[ -f docs/getting-started/installation.md ]]; then
    # Remote URL: .../download/vX.Y.Z/dress-graph-X.Y.Z.tar.gz
    sedi -E 's|/download/v[^/]+/dress-graph-[^"]+\.tar\.gz|/download/v'"${NEW}"'/dress-graph-'"${NEW}"'.tar.gz|g' \
        docs/getting-started/installation.md
    # Source: pkg install dress-graph-X.Y.Z.tar.gz
    sedi -E 's|dress-graph-[0-9]+\.[0-9]+\.[0-9]+\.tar\.gz|dress-graph-'"${NEW}"'.tar.gz|g' \
        docs/getting-started/installation.md
    echo "  ✓ docs/getting-started/installation.md"
    CHANGED=$((CHANGED + 1))
fi

# Conda — meta.yaml (in-repo recipe)
update conda/meta.yaml \
    '^\{% set version = ".*" %\}' \
    "{% set version = \"$NEW\" %}"

# vcpkg — vcpkg.json
update vcpkg/vcpkg.json \
    '"version": ".*"' \
    "\"version\": \"$NEW\""

# run_examples.sh — Octave tarball version
update run_examples.sh \
    'dress-graph-[0-9]+\.[0-9]+\.[0-9]+\.tar\.gz' \
    "dress-graph-${NEW}.tar.gz"

echo
echo "Updated $CHANGED file(s)."
