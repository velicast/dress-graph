# Installation

This guide covers installation for all supported languages.

## Python

### From PyPI (Remote)

```bash
pip install dress-graph
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

### vcpkg (Overlay Port)

If you use `vcpkg`, you can use the `vcpkg/` directory in this repository as an overlay port.

```bash
vcpkg install dress-graph --overlay-ports=/path/to/dress-graph/vcpkg
```

## MATLAB / Octave

### From Remote (Octave)

```octave
pkg install "https://github.com/velicast/dress-graph/releases/download/v0.3.1/dress-graph-0.3.1.tar.gz"
```

### From Source

Add the `matlab` directory to your MATLAB/Octave path.

**MATLAB:**
```matlab
addpath('path/to/dress-graph/matlab');
savepath;
```

**Octave:**
```octave
addpath('path/to/dress-graph/matlab');
savepath;
```
