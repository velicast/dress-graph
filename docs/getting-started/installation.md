# Installation

This guide covers installation for all supported languages.

---

## Prerequisites

### Core (CPU)

| Requirement | Version |
|---|---|
| C compiler | GCC or Clang with C11 support |
| CMake | ≥ 3.14 |
| OpenMP | Optional, auto-detected by CMake |

A C++17 compiler is additionally required for the `libdress++` (C++) wrapper.

### CUDA (GPU acceleration)

| Requirement | Details |
|---|---|
| NVIDIA GPU | Compute-capable device |
| CUDA Toolkit | Provides `nvcc` and `libcudart` |

CUDA is built separately from the core library, producing `libdress_cuda.so`.
Build it with:

```bash
./build.sh cuda
# or manually:
make -C libdress/src/cuda
```

### MPI (distributed computing)

| Requirement | Details |
|---|---|
| MPI implementation | OpenMPI or MPICH |
| `mpicc` | Must be on `PATH` |

For C / C++ via CMake, pass `-DDRESS_MPI=ON`:

```bash
cmake -S . -B build -DDRESS_MPI=ON
cmake --build build
```

The CUDA Makefile auto-detects `mpicc` and includes MPI+CUDA support when available.

### Per-language extras

| Language | CUDA requirement | MPI requirement |
|---|---|---|
| **C / C++** | Link against `libdress_cuda.so` + `-lcudart` | CMake flag `-DDRESS_MPI=ON` |
| **Python** | `libdress_cuda.so` on `LD_LIBRARY_PATH` | `pip install mpi4py`; run with `mpirun` |
| **Rust** | `libdress_cuda.so` on `LD_LIBRARY_PATH` | Cargo feature `mpi`; `mpicc` on PATH |
| **Go** | CGO: `libdress_cuda.so` + `-lcudart` | CGO: `-lmpi`; MPI headers on include path |
| **Julia** | `libdress_cuda.so` in `julia/` dir (or on path) | `MPI.jl` package; `libdress.so` with MPI symbols |
| **R** | `configure` auto-detects `libdress_cuda.so` | `configure` auto-detects `mpicc` |
| **Octave** | `libdress_cuda.so` on `LD_LIBRARY_PATH`; `nvcc` at build time | Not supported (CPU only) |
| **WASM** | Not supported (CPU only) | Not supported (CPU only) |

---

## Python

### From PyPI (Remote)

```bash
pip install dress-graph
```

### From conda-forge

```bash
conda install -c conda-forge dress-graph
```

### From Source

```bash
cd python
pip install .
```

To install in editable mode for development:

```bash
pip install -e .
```

## Rust

### From Crates.io (Remote)

```bash
cargo add dress-graph
```

### From Source

Add the local path to your `Cargo.toml`:

```toml
[dependencies]
dress-graph = { path = "path/to/dress-graph/rust" }
```

## JavaScript / WASM

### From npm (Remote)

```bash
npm install dress-graph
```

### From Source

Build the WASM package first:

```bash
./build.sh wasm
```

Then install from the local directory:

```bash
cd wasm
npm install .
```

Or link it:

```bash
cd wasm
npm link
# In your project:
npm link dress-graph
```

## Julia

### From GitHub (Remote)

```julia
using Pkg
Pkg.add(url="https://github.com/velicast/dress-graph", subdir="julia")
```

### From Source

```julia
using Pkg
Pkg.develop(path="julia")
```

## R

### From CRAN (Remote)

```r
install.packages("dress.graph")
```

### From GitHub (Remote)

```r
# install.packages("devtools")
devtools::install_github("velicast/dress-graph", subdir="r")
```

### From Source

```bash
R CMD INSTALL r
```

Or from within R:

```r
install.packages("./r", repos = NULL, type = "source")
```

## Go

### From GitHub (Remote)

```bash
go get github.com/velicast/dress-graph/go
```

### From Source

If you have the repository cloned locally, you can use a `replace` directive in your `go.mod`:

```bash
go mod edit -replace github.com/velicast/dress-graph/go=../path/to/dress-graph/go
```

## C / C++

### Homebrew (macOS / Linux)

```bash
brew tap velicast/dress-graph
brew install dress-graph
```

### vcpkg (Overlay Port)

If you use `vcpkg`, you can use the `vcpkg/` directory in this repository as an overlay port.

```bash
vcpkg install dress-graph --overlay-ports=/path/to/dress-graph/vcpkg
```

### From Source (CMake)

You can build the library using CMake directly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install
```

### From Source (Script)

The provided build script handles `libdress` (C), `libdress++` (C++), and `libdress-igraph`.

```bash
./build.sh c cpp igraph
```

## MATLAB / Octave

### From Remote (Octave)

```octave
pkg install "https://github.com/velicast/dress-graph/releases/download/v0.5.3/dress-graph-0.5.3.tar.gz"
```

### From Source (Octave)

Build the tarball and install locally:

```bash
./build.sh octave
```

```octave
pkg install dress-graph-0.5.3.tar.gz
pkg load dress-graph
```

### From Source (MATLAB)

Add the `matlab` directory to your MATLAB path:

```matlab
addpath('path/to/dress-graph/matlab');
savepath;
```
