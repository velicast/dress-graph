#!/usr/bin/env bash
# Publish all dress-graph packages to their respective registries.
#
# Usage:
#   ./publish.sh [FLAGS] [TARGET...]
#
# Targets: pypi npm cargo cran julia go octave brew vcpkg all (default: all)
#
# Flags:
#   --build-only       Build packages but do not upload to registries.
#   --install-local    Build and install packages locally (for testing).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# GitHub repo for the main project, the Homebrew tap, and the Julia package
GH_REPO="velicast/dress-graph"
GH_TAP_REPO="velicast/homebrew-dress-graph"
GH_JULIA_REPO="velicast/DRESS.jl"

# ── Parse flags ─────────────────────────────────────────────────────
BUILD_ONLY=0
INSTALL_LOCAL=0
TARGETS=()
for arg in "$@"; do
    case "$arg" in
        --build-only)     BUILD_ONLY=1 ;;
        --install-local)  BUILD_ONLY=1; INSTALL_LOCAL=1 ;;
        *)                TARGETS+=("$arg") ;;
    esac
done
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(all)
fi

publish_pypi() {
    echo "=== PyPI ==="
    cd "$ROOT/python"

    # Detect version from pyproject.toml (single source of truth)
    VERSION=$(grep -oP '^version = "\K[^"]+' "$ROOT/python/pyproject.toml")
    if [[ -z "$VERSION" ]]; then
        echo "ERROR: could not detect version from python/pyproject.toml"
        exit 1
    fi
    echo "  Version: $VERSION"

    # Use venv if available, otherwise system python3
    local PY="python3"
    local VENV="${VIRTUAL_ENV:-}"
    if [[ -z "$VENV" ]]; then
        for candidate in "$ROOT/.venv" "$ROOT/../.venv"; do
            if [[ -f "$candidate/bin/activate" ]]; then
                VENV="$(cd "$candidate" && pwd)"
                break
            fi
        done
    fi
    if [[ -n "$VENV" && -f "$VENV/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
        PY=python
    fi

    rm -rf dist build *.egg-info src/*.egg-info

    # Vendor libdress C/C++ sources for the native extension + CUDA auto-build
    local VENDOR="src/dress/_vendored"
    mkdir -p "$VENDOR/include/dress/cuda" "$VENDOR/include/dress/omp" "$VENDOR/src/cuda" "$VENDOR/src/mpi" "$VENDOR/src/omp"
    cp "$ROOT/libdress/include/dress/dress.h"           "$VENDOR/include/dress/"
    cp "$ROOT/libdress++/include/dress/dress.hpp"       "$VENDOR/include/dress/"
    cp "$ROOT/libdress/include/dress/cuda/dress_cuda.h" "$VENDOR/include/dress/cuda/"
    cp "$ROOT/libdress/include/dress/omp/dress_omp.h"   "$VENDOR/include/dress/omp/"
    cp "$ROOT/libdress/include/dress/omp/dress.h"       "$VENDOR/include/dress/omp/"
    cp "$ROOT/libdress/src/dress.c"                     "$VENDOR/src/"
    cp "$ROOT/libdress/src/delta_dress.c"               "$VENDOR/src/"
    cp "$ROOT/libdress/src/delta_dress_impl.c"          "$VENDOR/src/"
    cp "$ROOT/libdress/src/delta_dress_impl.h"          "$VENDOR/src/"
    cp "$ROOT/libdress/src/nabla_dress.c"               "$VENDOR/src/"
    cp "$ROOT/libdress/src/nabla_dress_impl.c"          "$VENDOR/src/"
    cp "$ROOT/libdress/src/nabla_dress_impl.h"          "$VENDOR/src/"
    cp "$ROOT/libdress/src/dress_histogram.c"           "$VENDOR/src/"
    cp "$ROOT/libdress/src/dress_histogram.h"           "$VENDOR/src/"
    cp "$ROOT/libdress/src/omp/dress_omp.c"             "$VENDOR/src/omp/"
    cp "$ROOT/libdress/src/omp/delta_dress_omp.c"       "$VENDOR/src/omp/"
    cp "$ROOT/libdress/src/omp/nabla_dress_omp.c"       "$VENDOR/src/omp/"
    cp "$ROOT/libdress/src/cuda/dress_cuda.cu"          "$VENDOR/src/cuda/"
    cp "$ROOT/libdress/src/cuda/delta_dress_cuda.c"     "$VENDOR/src/cuda/"
    cp "$ROOT/libdress/src/cuda/nabla_dress_cuda.c"     "$VENDOR/src/cuda/"
    cp "$ROOT/libdress/src/mpi/dress_mpi.c"             "$VENDOR/src/mpi/"

    $PY -m build
    echo "  ✓ wheel built: $(ls dist/*.whl)"

    # Cleanup vendored sources
    rm -rf "$VENDOR"

    if [[ $INSTALL_LOCAL -eq 1 ]]; then
        # Uninstall first to remove any editable-install .pth files that
        # would shadow the wheel and redirect imports to the source tree.
        $PY -m pip uninstall -y dress-graph 2>/dev/null || true
        $PY -m pip install --no-deps dist/*.whl

        # Refresh the installed vendored sources so local CUDA auto-builds
        # use the current workspace sources even when pip leaves stale files.
        local site=$($PY -c "import sysconfig; print(sysconfig.get_path('purelib'))")
        local site_vendor="$site/dress/_vendored"
        rm -rf "$site_vendor"
        mkdir -p "$site_vendor/include/dress/cuda" "$site_vendor/include/dress/omp" "$site_vendor/src/cuda" "$site_vendor/src/mpi" "$site_vendor/src/omp"
        cp "$ROOT/libdress/include/dress/dress.h"           "$site_vendor/include/dress/"
        cp "$ROOT/libdress++/include/dress/dress.hpp"       "$site_vendor/include/dress/"
        cp "$ROOT/libdress/include/dress/cuda/dress_cuda.h" "$site_vendor/include/dress/cuda/"
        cp "$ROOT/libdress/include/dress/omp/dress_omp.h"   "$site_vendor/include/dress/omp/"
        cp "$ROOT/libdress/include/dress/omp/dress.h"       "$site_vendor/include/dress/omp/"
        cp "$ROOT/libdress/src/dress.c"                     "$site_vendor/src/"
        cp "$ROOT/libdress/src/delta_dress.c"               "$site_vendor/src/"
        cp "$ROOT/libdress/src/delta_dress_impl.c"          "$site_vendor/src/"
        cp "$ROOT/libdress/src/delta_dress_impl.h"          "$site_vendor/src/"
        cp "$ROOT/libdress/src/dress_histogram.c"           "$site_vendor/src/"
        cp "$ROOT/libdress/src/dress_histogram.h"           "$site_vendor/src/"
        cp "$ROOT/libdress/src/omp/dress_omp.c"             "$site_vendor/src/omp/"
        cp "$ROOT/libdress/src/omp/delta_dress_omp.c"       "$site_vendor/src/omp/"
        cp "$ROOT/libdress/src/cuda/dress_cuda.cu"          "$site_vendor/src/cuda/"
        cp "$ROOT/libdress/src/cuda/delta_dress_cuda.c"     "$site_vendor/src/cuda/"
        cp "$ROOT/libdress/src/mpi/dress_mpi.c"             "$site_vendor/src/mpi/"

        # Remove stale auto-built artifacts so the CUDA module rebuilds
        # from the refreshed vendored sources on next import.
        rm -f "$site_vendor/src/cuda/"*.so "$site_vendor/src/cuda/"*.o 2>/dev/null || true
        echo "  ✓ installed locally"
    fi

    if [[ $BUILD_ONLY -eq 0 ]]; then
        # Publish via CI: create a GitHub release which triggers the
        # build_wheels.yml workflow (cross-platform cibuildwheel + trusted
        # publishing to PyPI).  No local API token or twine needed.

        if ! command -v gh &>/dev/null; then
            echo "ERROR: gh CLI is required to create a GitHub release."
            echo "  Install: https://cli.github.com"
            exit 1
        fi

        local TAG="v${VERSION}"

        # Ensure the working tree is clean
        cd "$ROOT"
        if ! git diff --quiet HEAD 2>/dev/null; then
            echo "ERROR: uncommitted changes in working tree.  Commit or stash first."
            exit 1
        fi

        # Create and push tag if it doesn't exist
        if ! git tag -l "$TAG" | grep -q .; then
            git tag "$TAG"
            echo "  ✓ created tag $TAG"
        fi
        git push origin "$TAG"
        echo "  ✓ pushed tag $TAG"

        # Create GitHub release (triggers build_wheels.yml → PyPI)
        if gh release view "$TAG" --repo "$GH_REPO" &>/dev/null; then
            echo "  Release $TAG already exists — CI may already be running."
        else
            gh release create "$TAG" --repo "$GH_REPO" \
                --title "dress-graph $VERSION" \
                --generate-notes
            echo "  ✓ created release $TAG"
        fi

        echo "  ✓ CI triggered: cross-platform wheels will be built and published to PyPI."
        echo "  Monitor: https://github.com/${GH_REPO}/actions"
        echo "=== PyPI (via CI) ==="
    fi
}

publish_npm() {
    echo "=== npm ==="
    cd "$ROOT/wasm"
    npm pack
    echo "  ✓ tarball built: $(ls *.tgz)"

    if [[ $INSTALL_LOCAL -eq 1 ]]; then
        cd "$ROOT/examples/wasm"
        npm install "$ROOT/wasm"/*.tgz
        echo "  ✓ installed locally in examples/wasm"
    fi

    if [[ $BUILD_ONLY -eq 0 ]]; then
        npm publish
        echo "=== npm uploaded ==="
    fi
}

publish_cargo() {
    echo "=== crates.io ==="
    # Vendor C sources into the crate (not committed to git)
    mkdir -p "$ROOT/rust/vendor/include/dress/cuda" "$ROOT/rust/vendor/include/dress/omp" "$ROOT/rust/vendor/include/dress/mpi" "$ROOT/rust/vendor/mpi"
    cp "$ROOT/libdress/src/dress.c"                "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/src/delta_dress.c"          "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/src/delta_dress_impl.c"     "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/src/delta_dress_impl.h"     "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/src/dress_histogram.c"      "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/src/dress_histogram.h"      "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/src/omp/dress_omp.c"        "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/src/omp/delta_dress_omp.c"  "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/src/nabla_dress.c"          "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/src/nabla_dress_impl.c"     "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/src/nabla_dress_impl.h"     "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/src/omp/nabla_dress_omp.c"  "$ROOT/rust/vendor/"
    cp "$ROOT/libdress/include/dress/dress.h"       "$ROOT/rust/vendor/include/dress/"
    # OMP headers
    cp "$ROOT/libdress/include/dress/omp/dress_omp.h" "$ROOT/rust/vendor/include/dress/omp/"
    cp "$ROOT/libdress/include/dress/omp/dress.h"     "$ROOT/rust/vendor/include/dress/omp/"
    # CUDA headers
    cp "$ROOT/libdress/include/dress/cuda/dress_cuda.h" "$ROOT/rust/vendor/include/dress/cuda/"
    cp "$ROOT/libdress/include/dress/cuda/dress.h"      "$ROOT/rust/vendor/include/dress/cuda/"
    # CUDA C source + pre-compiled kernel object
    cp "$ROOT/libdress/src/cuda/delta_dress_cuda.c"     "$ROOT/rust/vendor/delta_dress_cuda.c"
    if command -v nvcc &>/dev/null; then
        local cuda_o="$ROOT/libdress/src/cuda/dress_cuda.o"
        if [[ ! -f "$cuda_o" ]]; then
            nvcc -O2 -Xcompiler -fPIC -I"$ROOT/libdress/include" \
                -c "$ROOT/libdress/src/cuda/dress_cuda.cu" -o "$cuda_o"
        fi
        cp "$cuda_o" "$ROOT/rust/vendor/dress_cuda.o"
    fi
    # MPI sources + headers
    cp "$ROOT/libdress/src/mpi/dress_mpi.c"             "$ROOT/rust/vendor/mpi/"
    cp "$ROOT/libdress/include/dress/mpi/dress_mpi.h"   "$ROOT/rust/vendor/include/dress/mpi/"
    cp "$ROOT/libdress/include/dress/mpi/dress.h"       "$ROOT/rust/vendor/include/dress/mpi/" 2>/dev/null || true
    cp "$ROOT/LICENSE"                              "$ROOT/rust/LICENSE"

    (cd "$ROOT/rust" && cargo build --release 2>&1)
    echo "  ✓ crate built"

    if [[ $BUILD_ONLY -eq 0 ]]; then
        # Publish from /tmp to avoid WSL/NTFS metadata issues
        TMPDIR=$(mktemp -d)
        cp -r "$ROOT/rust/." "$TMPDIR/"
        cd "$TMPDIR"
        cargo publish
        rm -rf "$TMPDIR"
        echo "=== crates.io uploaded ==="
    fi

    # Keep vendor/ when --install-local; cleanup otherwise
    if [[ $INSTALL_LOCAL -eq 0 && $BUILD_ONLY -eq 0 ]]; then
        rm -rf "$ROOT/rust/vendor" "$ROOT/rust/LICENSE"
    fi
}

publish_cran() {
    echo "=== R/CRAN ==="
    # Vendor C sources into r/src/ (not committed to git)
    cp "$ROOT/libdress/src/dress.c"                "$ROOT/r/src/"
    cp "$ROOT/libdress/src/delta_dress.c"          "$ROOT/r/src/"
    cp "$ROOT/libdress/src/delta_dress_impl.c"     "$ROOT/r/src/"
    cp "$ROOT/libdress/src/delta_dress_impl.h"     "$ROOT/r/src/"
    cp "$ROOT/libdress/src/dress_histogram.c"      "$ROOT/r/src/"
    cp "$ROOT/libdress/src/dress_histogram.h"      "$ROOT/r/src/"
    cp "$ROOT/libdress/src/nabla_dress.c"          "$ROOT/r/src/"
    cp "$ROOT/libdress/src/nabla_dress_impl.c"     "$ROOT/r/src/"
    cp "$ROOT/libdress/src/nabla_dress_impl.h"     "$ROOT/r/src/"
    mkdir -p "$ROOT/r/src/dress"
    cp "$ROOT/libdress/include/dress/dress.h"       "$ROOT/r/src/dress/"

    # OMP headers + sources
    mkdir -p "$ROOT/r/src/dress/omp"
    cp "$ROOT/libdress/include/dress/omp/dress_omp.h" "$ROOT/r/src/dress/omp/"
    cp "$ROOT/libdress/include/dress/omp/dress.h"     "$ROOT/r/src/dress/omp/"
    cp "$ROOT/libdress/src/omp/dress_omp.c"           "$ROOT/r/src/"
    cp "$ROOT/libdress/src/omp/delta_dress_omp.c"     "$ROOT/r/src/"
    sed -i 's|"../delta_dress_impl.h"|"delta_dress_impl.h"|' "$ROOT/r/src/dress_omp.c" 2>/dev/null || true
    sed -i 's|"../dress_histogram.h"|"dress_histogram.h"|' "$ROOT/r/src/dress_omp.c" 2>/dev/null || true
    sed -i 's|"../delta_dress_impl.h"|"delta_dress_impl.h"|' "$ROOT/r/src/delta_dress_omp.c" 2>/dev/null || true
    sed -i 's|"../dress_histogram.h"|"dress_histogram.h"|' "$ROOT/r/src/delta_dress_omp.c" 2>/dev/null || true
    cp "$ROOT/libdress/src/omp/nabla_dress_omp.c"     "$ROOT/r/src/"
    sed -i 's|"../nabla_dress_impl.h"|"nabla_dress_impl.h"|' "$ROOT/r/src/nabla_dress_omp.c" 2>/dev/null || true
    sed -i 's|"../dress_histogram.h"|"dress_histogram.h"|' "$ROOT/r/src/nabla_dress_omp.c" 2>/dev/null || true

    # CUDA headers (if available)
    if [ -d "$ROOT/libdress/include/dress/cuda" ]; then
        mkdir -p "$ROOT/r/src/dress/cuda"
        cp "$ROOT/libdress/include/dress/cuda/"*.h "$ROOT/r/src/dress/cuda/"
    fi
    # CUDA sources (for compile-from-source at install time)
    if [ -f "$ROOT/libdress/src/cuda/delta_dress_cuda.c" ]; then
        cp "$ROOT/libdress/src/cuda/delta_dress_cuda.c" "$ROOT/r/src/"
    fi
    if [ -f "$ROOT/libdress/src/cuda/nabla_dress_cuda.c" ]; then
        cp "$ROOT/libdress/src/cuda/nabla_dress_cuda.c" "$ROOT/r/src/"
        sed -i 's|"../nabla_dress_impl.h"|"nabla_dress_impl.h"|' "$ROOT/r/src/nabla_dress_cuda.c" 2>/dev/null || true
        sed -i 's|"../dress_histogram.h"|"dress_histogram.h"|' "$ROOT/r/src/nabla_dress_cuda.c" 2>/dev/null || true
    fi
    if [ -f "$ROOT/libdress/src/cuda/dress_cuda.cu" ]; then
        cp "$ROOT/libdress/src/cuda/dress_cuda.cu" "$ROOT/r/src/"
    fi

    # MPI headers + source (if available)
    if [ -d "$ROOT/libdress/include/dress/mpi" ]; then
        mkdir -p "$ROOT/r/src/dress/mpi"
        cp "$ROOT/libdress/include/dress/mpi/"*.h "$ROOT/r/src/dress/mpi/"
    fi
    if [ -f "$ROOT/libdress/src/mpi/dress_mpi.c" ]; then
        cp "$ROOT/libdress/src/mpi/dress_mpi.c" "$ROOT/r/src/"
        sed -i 's|"../delta_dress_impl.h"|"delta_dress_impl.h"|' "$ROOT/r/src/dress_mpi.c"
        sed -i 's|"../dress_histogram.h"|"dress_histogram.h"|' "$ROOT/r/src/dress_mpi.c"
        sed -i 's|"../nabla_dress_impl.h"|"nabla_dress_impl.h"|' "$ROOT/r/src/dress_mpi.c"
    fi

    cd "$ROOT"
    R CMD build r/
    local tarball
    tarball=$(ls -t dress.graph_*.tar.gz 2>/dev/null | head -1)
    echo "  ✓ R tarball built: $tarball"

    if [[ $INSTALL_LOCAL -eq 1 ]]; then
        local R_USER_LIB="${R_LIBS_USER:-$HOME/R/library}"
        mkdir -p "$R_USER_LIB"
        R CMD INSTALL --library="$R_USER_LIB" "$tarball"
        echo "  ✓ installed locally to $R_USER_LIB"
    fi

    if [[ $BUILD_ONLY -eq 0 ]]; then
        echo "  Submit $tarball at https://cran.r-project.org/submit.html"
    fi

    # Cleanup vendored sources
    rm -f  "$ROOT/r/src/dress.c" "$ROOT/r/src/delta_dress.c" "$ROOT/r/src/delta_dress_impl.c" "$ROOT/r/src/delta_dress_impl.h"
    rm -f  "$ROOT/r/src/dress_histogram.c" "$ROOT/r/src/dress_histogram.h"
    rm -f  "$ROOT/r/src/dress_omp.c" "$ROOT/r/src/delta_dress_omp.c"
    rm -f  "$ROOT/r/src/dress_mpi.c" "$ROOT/r/src/delta_dress_cuda.c" "$ROOT/r/src/dress_cuda.cu" "$ROOT/r/src/dress_cuda.o"
    rm -rf "$ROOT/r/src/dress"
}

publish_brew() {
    echo "=== Homebrew ==="

    if [[ $BUILD_ONLY -eq 1 ]]; then
        echo "  (skipped — brew tap requires a GitHub release)"
        return
    fi

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
        p_dress_graph_t g = dress_init_graph(3, 3, U, V, NULL, 0, 0);
        if (!g) return 1;
        printf("OK\\\\n");
        dress_free_graph(g);
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
    echo "=== Julia ==="

    # Detect version from Project.toml
    VERSION=$(grep -oP '^version = "\K[^"]+' "$ROOT/julia/Project.toml")
    if [[ -z "$VERSION" ]]; then
        echo "ERROR: could not detect version from julia/Project.toml"
        exit 1
    fi
    echo "  Version: $VERSION"

    # Vendor C sources into julia/vendor/ for standalone use
    local VENDOR="$ROOT/julia/vendor"
    mkdir -p "$VENDOR/include/dress/cuda" "$VENDOR/include/dress/mpi" "$VENDOR/include/dress/omp" \
             "$VENDOR/src/cuda" "$VENDOR/src/mpi" "$VENDOR/src/omp"
    cp "$ROOT/libdress/include/dress/dress.h"           "$VENDOR/include/dress/"
    cp "$ROOT/libdress/include/dress/omp/dress_omp.h"   "$VENDOR/include/dress/omp/"
    cp "$ROOT/libdress/include/dress/omp/dress.h"       "$VENDOR/include/dress/omp/"
    cp "$ROOT/libdress/src/dress.c"                     "$VENDOR/src/"
    cp "$ROOT/libdress/src/delta_dress.c"               "$VENDOR/src/"
    cp "$ROOT/libdress/src/delta_dress_impl.c"          "$VENDOR/src/"
    cp "$ROOT/libdress/src/delta_dress_impl.h"          "$VENDOR/src/"
    cp "$ROOT/libdress/src/dress_histogram.c"           "$VENDOR/src/"
    cp "$ROOT/libdress/src/dress_histogram.h"           "$VENDOR/src/"
    cp "$ROOT/libdress/src/omp/dress_omp.c"             "$VENDOR/src/omp/"
    cp "$ROOT/libdress/src/omp/delta_dress_omp.c"       "$VENDOR/src/omp/"
    cp "$ROOT/libdress/src/nabla_dress.c"               "$VENDOR/src/"
    cp "$ROOT/libdress/src/nabla_dress_impl.c"          "$VENDOR/src/"
    cp "$ROOT/libdress/src/nabla_dress_impl.h"          "$VENDOR/src/"
    cp "$ROOT/libdress/src/omp/nabla_dress_omp.c"       "$VENDOR/src/omp/"
    echo "  ✓ C sources vendored into julia/vendor/"

    if [[ $INSTALL_LOCAL -eq 1 ]]; then
        julia -e "using Pkg; Pkg.develop(path=\"$ROOT/julia\")"
        echo "  ✓ installed locally via Pkg.develop"
    fi

    if [[ $BUILD_ONLY -eq 1 ]]; then
        echo "  ✓ Julia package ready at $ROOT/julia"
        return
    fi

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
    # Cleanup vendored sources from the local tree
    rm -rf "$VENDOR"
    echo "=== Julia done: tag v${VERSION} pushed to ${GH_JULIA_REPO} ==="
    echo "  To register: comment '@JuliaRegistrator register' on the tagged commit."
}

publish_go() {
    echo "=== Go ==="

    VERSION=$(grep -oP '^version = "\K[^"]+' "$ROOT/python/pyproject.toml")
    if [[ -z "$VERSION" ]]; then
        echo "ERROR: could not detect version from python/pyproject.toml"
        exit 1
    fi
    echo "  Version: $VERSION"

    # Vendor C sources into each Go sub-module
    _vendor_go_sources() {
        local DEST="$1"
        mkdir -p "$DEST/include/dress/cuda" "$DEST/include/dress/mpi" \
                 "$DEST/include/dress/omp" \
                 "$DEST/src/cuda" "$DEST/src/mpi" "$DEST/src/omp"
        # Headers
        cp "$ROOT/libdress/include/dress/dress.h"            "$DEST/include/dress/"
        cp "$ROOT/libdress/include/dress/cuda/dress_cuda.h"  "$DEST/include/dress/cuda/" 2>/dev/null || true
        cp "$ROOT/libdress/include/dress/omp/dress_omp.h"    "$DEST/include/dress/omp/"
        cp "$ROOT/libdress/include/dress/omp/dress.h"        "$DEST/include/dress/omp/"
        cp "$ROOT/libdress/include/dress/mpi/dress_mpi.h"    "$DEST/include/dress/mpi/"  2>/dev/null || true
        # C sources
        cp "$ROOT/libdress/src/dress.c"                      "$DEST/src/"
        cp "$ROOT/libdress/src/dress_histogram.c"            "$DEST/src/"
        cp "$ROOT/libdress/src/dress_histogram.h"            "$DEST/src/"
        cp "$ROOT/libdress/src/delta_dress.c"                "$DEST/src/"
        cp "$ROOT/libdress/src/delta_dress_impl.c"           "$DEST/src/"
        cp "$ROOT/libdress/src/delta_dress_impl.h"           "$DEST/src/"
        cp "$ROOT/libdress/src/nabla_dress.c"                  "$DEST/src/"
        cp "$ROOT/libdress/src/nabla_dress_impl.c"             "$DEST/src/"
        cp "$ROOT/libdress/src/nabla_dress_impl.h"             "$DEST/src/"
        # OMP sources
        cp "$ROOT/libdress/src/omp/dress_omp.c"              "$DEST/src/omp/"
        cp "$ROOT/libdress/src/omp/delta_dress_omp.c"        "$DEST/src/omp/"
        cp "$ROOT/libdress/src/omp/nabla_dress_omp.c"        "$DEST/src/omp/"
        # CUDA sources
        cp "$ROOT/libdress/src/cuda/delta_dress_cuda.c"      "$DEST/src/cuda/"  2>/dev/null || true
        cp "$ROOT/libdress/src/cuda/nabla_dress_cuda.c"      "$DEST/src/cuda/"  2>/dev/null || true
        # MPI sources
        cp "$ROOT/libdress/src/mpi/dress_mpi.c"              "$DEST/src/mpi/"   2>/dev/null || true
    }

    for mod_dir in "$ROOT/go" "$ROOT/go/omp" "$ROOT/go/cuda" "$ROOT/go/mpi" "$ROOT/go/mpi/omp" "$ROOT/go/mpi/cuda"; do
        _vendor_go_sources "$mod_dir/vendor"
        echo "  ✓ vendored into $(basename "$(dirname "$mod_dir")")/$(basename "$mod_dir")/vendor/"
    done

    # CUDA: also link pre-compiled kernel if nvcc was available
    for cuda_dir in "$ROOT/go/cuda" "$ROOT/go/mpi/cuda"; do
        local cuda_o="$ROOT/libdress/src/cuda/dress_cuda.o"
        if [[ -f "$cuda_o" ]]; then
            mkdir -p "$cuda_dir/vendor/lib"
            # Build a static library from the kernel object
            ar rcs "$cuda_dir/vendor/lib/libdress_cuda.a" "$cuda_o"
        fi
    done

    if [[ $INSTALL_LOCAL -eq 1 ]]; then
        echo "  Go modules are imported directly — use the local path:"
        echo "    go mod edit -replace github.com/velicast/dress-graph/go=$ROOT/go"
        echo "  ✓ vendor/ populated for local builds"
    fi

    if [[ $BUILD_ONLY -eq 1 ]]; then
        echo "  ✓ Go modules ready with vendored sources"
        return
    fi

    # Tag and push each sub-module
    cd "$ROOT"
    TAG="v${VERSION}"
    GO_TAG="go/${TAG}"

    if ! git tag -l "$GO_TAG" | grep -q .; then
        git tag "$GO_TAG"
        echo "  ✓ created tag $GO_TAG"
    fi

    # Sub-module tags (required by Go module proxy)
    for sub in omp cuda mpi mpi/omp mpi/cuda; do
        SUB_TAG="go/${sub}/${TAG}"
        if ! git tag -l "$SUB_TAG" | grep -q .; then
            git tag "$SUB_TAG"
            echo "  ✓ created tag $SUB_TAG"
        fi
    done

    git push origin "$GO_TAG"
    for sub in omp cuda mpi mpi/omp mpi/cuda; do
        git push origin "go/${sub}/${TAG}"
    done

    # Cleanup vendored sources
    for mod_dir in "$ROOT/go" "$ROOT/go/omp" "$ROOT/go/cuda" "$ROOT/go/mpi" "$ROOT/go/mpi/omp" "$ROOT/go/mpi/cuda"; do
        rm -rf "$mod_dir/vendor"
    done

    echo "=== Go done: tags pushed ==="
    echo "  Users install with:  go get github.com/velicast/dress-graph/go@${TAG}"
}

publish_octave() {
    echo "=== Octave ==="

    if [[ $BUILD_ONLY -eq 1 ]]; then
        echo "  (skipped — octave tarball built by build.sh)"
        return
    fi

    # Detect version from DESCRIPTION
    VERSION=$(grep -oP '^Version: \K.*' "$ROOT/octave/DESCRIPTION")
    if [[ -z "$VERSION" ]]; then
        echo "ERROR: could not detect version from octave/DESCRIPTION"
        exit 1
    fi
    echo "  Version: $VERSION"

    TARBALL="$ROOT/dress-graph-${VERSION}.tar.gz"
    if [[ ! -f "$TARBALL" ]]; then
        echo "  Tarball not found — building octave package ..."
        bash "$ROOT/build.sh" octave
        if [[ ! -f "$TARBALL" ]]; then
            echo "ERROR: $TARBALL still not found after build.sh octave."
            exit 1
        fi
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
    echo "=== vcpkg ==="

    if [[ $BUILD_ONLY -eq 1 ]]; then
        echo "  (skipped — vcpkg port does not need building)"
        return
    fi

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

dispatch() {
    case "$1" in
        pypi)  publish_pypi  ;;
        npm)   publish_npm   ;;
        cargo) publish_cargo ;;
        cran)  publish_cran  ;;
        julia)  publish_julia  ;;
        go)     publish_go     ;;
        octave) publish_octave ;;
        brew)   publish_brew   ;;
        vcpkg)  publish_vcpkg  ;;
        all)
            publish_pypi
            publish_npm
            publish_cargo
            publish_cran
            publish_julia
            publish_go
            publish_octave
            publish_brew
            publish_vcpkg
            ;;
        *)
            echo "Unknown target: $1"
            echo "Usage: $0 [--build-only|--install-local] [pypi|npm|cargo|cran|julia|go|octave|brew|vcpkg|all]"
            exit 1
            ;;
    esac
}

for t in "${TARGETS[@]}"; do
    dispatch "$t"
done

echo "=== Done ==="
