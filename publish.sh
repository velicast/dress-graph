#!/usr/bin/env bash
# Publish all dress-graph packages to their respective registries.
# Usage: ./publish.sh [pypi|npm|cargo|cran|all]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-all}"

publish_pypi() {
    echo "=== Publishing to PyPI ==="
    cd "$ROOT/python"
    rm -rf dist build *.egg-info src/*.egg-info
    python -m build
    twine upload dist/*
    echo "=== PyPI done ==="
}

publish_npm() {
    echo "=== Publishing to npm ==="
    cd "$ROOT/wasm"
    npm publish
    echo "=== npm done ==="
}

publish_cargo() {
    echo "=== Publishing to crates.io ==="
    # Vendor C sources into the crate (not committed to git)
    mkdir -p "$ROOT/rust/vendor/include/dress"
    cp "$ROOT/libdress/src/dress.c"          "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/include/dress/dress.h" "$ROOT/rust/vendor/include/dress/"
    cp "$ROOT/LICENSE"                        "$ROOT/rust/LICENSE"

    # Publish from /tmp to avoid WSL/NTFS metadata issues
    TMPDIR=$(mktemp -d)
    cp -r "$ROOT/rust/." "$TMPDIR/"
    cd "$TMPDIR"
    cargo publish

    # Cleanup
    rm -rf "$TMPDIR"
    rm -rf "$ROOT/rust/vendor" "$ROOT/rust/LICENSE"
    echo "=== crates.io done ==="
}

publish_cran() {
    echo "=== Building R/CRAN package ==="
    # Vendor C sources into r/src/ (not committed to git)
    cp "$ROOT/libdress/src/dress.c" "$ROOT/r/src/"
    mkdir -p "$ROOT/r/src/dress"
    cp "$ROOT/libdress/include/dress/dress.h" "$ROOT/r/src/dress/"

    cd "$ROOT"
    R CMD build r/
    echo "=== R tarball built. Submit dress.graph_*.tar.gz at https://cran.r-project.org/submit.html ==="

    # Cleanup vendored sources
    rm -f  "$ROOT/r/src/dress.c"
    rm -rf "$ROOT/r/src/dress"
}

case "$TARGET" in
    pypi)  publish_pypi  ;;
    npm)   publish_npm   ;;
    cargo) publish_cargo ;;
    cran)  publish_cran  ;;
    all)
        publish_pypi
        publish_npm
        publish_cargo
        publish_cran
        echo "=== All packages published ==="
        ;;
    *)
        echo "Usage: $0 [pypi|npm|cargo|cran|all]"
        exit 1
        ;;
esac
