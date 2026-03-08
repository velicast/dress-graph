#!/usr/bin/env bash
# Publish all dress-graph packages to their respective registries.
# Usage: ./publish.sh [pypi|npm|cargo|cran|julia|octave|brew|vcpkg|all]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-all}"

# GitHub repo for the main project, the Homebrew tap, and the Julia package
GH_REPO="velicast/dress-graph"
GH_TAP_REPO="velicast/homebrew-dress-graph"
GH_JULIA_REPO="velicast/DRESS.jl"

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
    cp "$ROOT/libdress/src/dress.c"                "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/src/delta_dress.c"          "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/src/delta_dress_impl.h"     "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/include/dress/dress.h"       "$ROOT/rust/vendor/include/dress/"
    cp "$ROOT/libdress/include/dress/delta_dress.h" "$ROOT/rust/vendor/include/dress/"
    cp "$ROOT/LICENSE"                              "$ROOT/rust/LICENSE"

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
    cp "$ROOT/libdress/src/dress.c"                "$ROOT/r/src/"
    cp "$ROOT/libdress/src/delta_dress.c"          "$ROOT/r/src/"
    cp "$ROOT/libdress/src/delta_dress_impl.h"     "$ROOT/r/src/"
    mkdir -p "$ROOT/r/src/dress"
    cp "$ROOT/libdress/include/dress/dress.h"       "$ROOT/r/src/dress/"
    cp "$ROOT/libdress/include/dress/delta_dress.h" "$ROOT/r/src/dress/"

    cd "$ROOT"
    R CMD build r/
    echo "=== R tarball built. Submit dress.graph_*.tar.gz at https://cran.r-project.org/submit.html ==="

    # Cleanup vendored sources
    rm -f  "$ROOT/r/src/dress.c" "$ROOT/r/src/delta_dress.c" "$ROOT/r/src/delta_dress_impl.h"
    rm -rf "$ROOT/r/src/dress"
}

publish_brew() {
    echo "=== Publishing Homebrew tap ==="

    # Detect version from python/pyproject.toml (single source of truth)
    VERSION=$(grep -oP '^version = "\K[^"]+' "$ROOT/python/pyproject.toml")
    if [[ -z "$VERSION" ]]; then
        echo "ERROR: could not detect version from python/pyproject.toml"
        exit 1
    fi
    echo "  Version: $VERSION"

    TARBALL_URL="https://github.com/${GH_REPO}/archive/refs/tags/v${VERSION}.tar.gz"

    # Download tarball and compute SHA256
    echo "  Fetching $TARBALL_URL ..."
    SHA256=$(curl -sL "$TARBALL_URL" | sha256sum | awk '{print $1}')
    if [[ "$SHA256" == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" ]]; then
        echo "ERROR: got empty-file SHA256 — tag v${VERSION} may not exist on GitHub yet."
        echo "  Push the tag first:  git tag v${VERSION} && git push origin --tags"
        exit 1
    fi
    echo "  SHA256: $SHA256"

    # Clone (or update) the tap repo into /tmp
    TAP_DIR=$(mktemp -d)
    TAP_URL="https://github.com/${GH_TAP_REPO}.git"

    # Prefer gh CLI for auth; fall back to HTTPS (prompts for credentials)
    if command -v gh &>/dev/null; then
        # Ensure tap repo exists on GitHub
        if ! gh repo view "$GH_TAP_REPO" &>/dev/null; then
            echo "  Creating GitHub repo ${GH_TAP_REPO} ..."
            gh repo create "$GH_TAP_REPO" --public \
                --description "Homebrew tap for dress-graph"
        fi
        gh repo clone "$GH_TAP_REPO" "$TAP_DIR" -- --depth 1 2>/dev/null || {
            echo "  Initializing fresh tap repo ..."
            cd "$TAP_DIR"
            git init
            git remote add origin "$TAP_URL"
        }
    else
        if git ls-remote "$TAP_URL" &>/dev/null; then
            git clone --depth 1 "$TAP_URL" "$TAP_DIR"
        else
            echo "  ERROR: repo ${GH_TAP_REPO} does not exist and 'gh' CLI is not installed."
            echo "  Either install gh (https://cli.github.com) or create the repo manually:"
            echo "    https://github.com/new  →  ${GH_TAP_REPO}"
            rm -rf "$TAP_DIR"
            exit 1
        fi
    fi

    mkdir -p "$TAP_DIR/Formula"

    # Write the formula
    cat > "$TAP_DIR/Formula/dress-graph.rb" <<RUBY
class DressGraph < Formula
  desc "DRESS edge similarity for graphs — C/C++ library"
  homepage "https://github.com/${GH_REPO}"
  url "${TARBALL_URL}"
  sha256 "${SHA256}"
  license "MIT"

  depends_on "cmake" => :build

  def install
    system "cmake", "-S", ".", "-B", "build",
           *std_cmake_args,
           "-DBUILD_SHARED_LIBS=ON"
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  test do
    (testpath/"test.c").write <<~C
      #include <dress/dress.h>
      #include <stdio.h>
      int main() {
        int U[] = {0, 1, 2};
        int V[] = {1, 2, 0};
        p_dress_graph_t g = init_dress_graph(3, 3, U, V, NULL, 0, 0);
        if (!g) return 1;
        printf("OK\\\\n");
        free_dress_graph(g);
        return 0;
      }
    C
    system ENV.cc, "test.c", "-I#{include}", "-L#{lib}", "-ldress", "-lm", "-o", "test"
    assert_match "OK", shell_output("./test")
  end
end
RUBY

    # Commit and push
    cd "$TAP_DIR"
    git add -A
    git commit -m "dress-graph ${VERSION}"

    # Use gh CLI for authenticated push if available
    if command -v gh &>/dev/null; then
        REMOTE_URL="https://github.com/${GH_TAP_REPO}.git"
        git remote set-url origin "$REMOTE_URL"
        gh auth setup-git 2>/dev/null || true
    fi
    git push -u origin "$(git branch --show-current)"

    rm -rf "$TAP_DIR"
    echo "=== Homebrew tap done: brew tap ${GH_TAP_REPO/homebrew-/} && brew install dress-graph ==="
}

publish_julia() {
    echo "=== Publishing Julia package to DRESS.jl ==="

    # Detect version from Project.toml
    VERSION=$(grep -oP '^version = "\K[^"]+' "$ROOT/julia/Project.toml")
    if [[ -z "$VERSION" ]]; then
        echo "ERROR: could not detect version from julia/Project.toml"
        exit 1
    fi
    echo "  Version: $VERSION"

    # Clone the DRESS.jl repo into /tmp
    JULIA_DIR=$(mktemp -d)
    JULIA_URL="https://github.com/${GH_JULIA_REPO}.git"

    if command -v gh &>/dev/null; then
        gh repo clone "$GH_JULIA_REPO" "$JULIA_DIR" -- --depth 1 2>/dev/null || {
            cd "$JULIA_DIR"
            git init
            git remote add origin "$JULIA_URL"
        }
    else
        git clone --depth 1 "$JULIA_URL" "$JULIA_DIR"
    fi

    # Sync package contents
    cp "$ROOT/julia/Project.toml" "$JULIA_DIR/"
    cp "$ROOT/julia/README.md"    "$JULIA_DIR/" 2>/dev/null || true
    rm -rf "$JULIA_DIR/src" "$JULIA_DIR/test"
    cp -r "$ROOT/julia/src"  "$JULIA_DIR/"
    cp -r "$ROOT/julia/test" "$JULIA_DIR/" 2>/dev/null || true
    cp "$ROOT/LICENSE"        "$JULIA_DIR/" 2>/dev/null || true

    # Commit, tag, and push
    cd "$JULIA_DIR"
    git add -A
    if git diff --cached --quiet; then
        echo "  No changes to commit."
    else
        git commit -m "DRESS v${VERSION}"
    fi

    # Tag if not already tagged
    if ! git tag -l "v${VERSION}" | grep -q .; then
        git tag "v${VERSION}"
    fi

    if command -v gh &>/dev/null; then
        gh auth setup-git 2>/dev/null || true
    fi
    git push -u origin "$(git branch --show-current)"
    git push origin "v${VERSION}"

    rm -rf "$JULIA_DIR"
    echo "=== Julia done: tag v${VERSION} pushed to ${GH_JULIA_REPO} ==="
    echo "  To register: comment '@JuliaRegistrator register' on the tagged commit."
}

publish_octave() {
    echo "=== Publishing Octave package (GitHub Release attachment) ==="

    # Detect version from DESCRIPTION
    VERSION=$(grep -oP '^Version: \K.*' "$ROOT/octave/DESCRIPTION")
    if [[ -z "$VERSION" ]]; then
        echo "ERROR: could not detect version from octave/DESCRIPTION"
        exit 1
    fi
    echo "  Version: $VERSION"

    TARBALL="$ROOT/dress-graph-${VERSION}.tar.gz"
    if [[ ! -f "$TARBALL" ]]; then
        echo "ERROR: $TARBALL not found. Run './build.sh octave' first."
        exit 1
    fi

    if ! command -v gh &>/dev/null; then
        echo "ERROR: gh CLI is required to attach assets to GitHub Releases."
        echo "  Install: https://cli.github.com"
        exit 1
    fi

    TAG="v${VERSION}"

    # Create the release if it doesn't exist yet
    if ! gh release view "$TAG" --repo "$GH_REPO" &>/dev/null; then
        echo "  Creating GitHub release $TAG ..."
        gh release create "$TAG" --repo "$GH_REPO" \
            --title "dress-graph $VERSION" \
            --notes "Release $VERSION"
    fi

    # Upload (overwrite if already attached)
    echo "  Attaching $TARBALL to release $TAG ..."
    gh release upload "$TAG" "$TARBALL" --repo "$GH_REPO" --clobber

    echo "  ✓ Attached to https://github.com/${GH_REPO}/releases/tag/${TAG}"
    echo "  Users install with:"
    echo "    pkg install \"https://github.com/${GH_REPO}/releases/download/${TAG}/dress-graph-${VERSION}.tar.gz\""
    echo "=== Octave done ==="
}

publish_vcpkg() {
    echo "=== Publishing vcpkg port ==="

    # Detect version
    VERSION=$(grep -oP '^version = "\K[^"]+' "$ROOT/python/pyproject.toml")
    if [[ -z "$VERSION" ]]; then
        echo "ERROR: could not detect version from python/pyproject.toml"
        exit 1
    fi
    echo "  Version: $VERSION"

    TARBALL_URL="https://github.com/${GH_REPO}/archive/refs/tags/v${VERSION}.tar.gz"

    # Download tarball and compute SHA512
    echo "  Fetching $TARBALL_URL ..."
    SHA512=$(curl -sL "$TARBALL_URL" | sha512sum | awk '{print $1}')
    EMPTY_SHA512="cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
    if [[ "$SHA512" == "$EMPTY_SHA512" ]]; then
        echo "ERROR: got empty-file SHA512 — tag v${VERSION} may not exist on GitHub yet."
        exit 1
    fi
    echo "  SHA512: $SHA512"

    # Update portfile.cmake with correct REF and SHA512
    sed -i -E "s|REF \"v[^\"]*\"|REF \"v${VERSION}\"|" "$ROOT/vcpkg/portfile.cmake"
    sed -i -E "s|SHA512 .*|SHA512 ${SHA512}|" "$ROOT/vcpkg/portfile.cmake"

    # Update vcpkg.json version
    sed -i -E "s|\"version\": \"[^\"]*\"|\"version\": \"${VERSION}\"|" "$ROOT/vcpkg/vcpkg.json"

    echo "  ✓ Updated vcpkg/portfile.cmake and vcpkg/vcpkg.json"
    echo
    echo "  To submit to vcpkg registry, copy vcpkg/ to a vcpkg fork and open a PR:"
    echo "    cp -r vcpkg/ <vcpkg-clone>/ports/dress-graph/"
    echo "    cd <vcpkg-clone> && git add ports/dress-graph && git commit && gh pr create"
    echo
    echo "  For local/overlay use:"
    echo "    vcpkg install dress-graph --overlay-ports=$ROOT/vcpkg"
    echo "=== vcpkg done ==="
}

case "$TARGET" in
    pypi)  publish_pypi  ;;
    npm)   publish_npm   ;;
    cargo) publish_cargo ;;
    cran)  publish_cran  ;;
    julia)  publish_julia  ;;
    octave) publish_octave ;;
    brew)   publish_brew   ;;
    vcpkg)  publish_vcpkg  ;;
    all)
        publish_pypi
        publish_npm
        publish_cargo
        publish_cran
        publish_julia
        publish_octave
        publish_brew
        publish_vcpkg
        echo "=== All packages published ==="
        ;;
    *)
        echo "Usage: $0 [pypi|npm|cargo|cran|julia|octave|brew|vcpkg|all]"
        exit 1
        ;;
esac
