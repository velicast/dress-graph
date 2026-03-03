# Installation

## Python (recommended)

```bash
pip install dress-graph
```

### Conda (coming soon)

```bash
conda install -c conda-forge dress-graph
```

!!! note "Not yet available"
    The conda-forge package is pending review. Use `pip install dress-graph` in the meantime.

Or build from source:

```bash
cd python
pip install .
```

## Rust

Add to `Cargo.toml`:

```bash
cargo add dress-graph
```

Or add to `Cargo.toml` manually:

```toml
[dependencies]
dress-graph = "0.1"
```

## C / C++

### Homebrew (macOS / Linux)

```bash
brew tap velicast/dress-graph
brew install dress-graph
```

### vcpkg (overlay port)

```bash
# Clone dress-graph and use its vcpkg/ directory as an overlay
vcpkg install dress-graph --overlay-ports=/path/to/dress-graph/vcpkg
```

!!! note "Not yet in vcpkg registry"
    Until the port is accepted into the official vcpkg registry,
    use the `--overlay-ports` flag pointing to the `vcpkg/` directory
    in this repo.

### From source

```bash
./build.sh c cpp
```

This builds `libdress` (static and shared) and the header-only C++ wrapper.

## Go

```go
import "github.com/velicast/dress-graph/go"
```

Requires CGo and a C compiler.

## JavaScript / WASM

```bash
npm install dress-graph
```

Or build from source:

```bash
./build.sh wasm
```

Then in Node.js:

```javascript
import { dressFit, Variant } from 'dress-graph';
```

## Julia

```julia
# From the repo root (uses Project.toml)
julia --project=julia
using DRESS
```

## R

```r
# From source
R CMD INSTALL r/
```

## MATLAB

```matlab
cd matlab
dress_build   % compiles the MEX file
```

## Octave

```octave
pkg install "https://github.com/velicast/dress-graph/releases/download/v0.3.1/dress-graph-0.3.1.tar.gz"
pkg load dress-graph
```
