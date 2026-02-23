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

```bash
mkdir build && cd build
cmake ..
make
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
cd wasm
bash build.sh
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

## MATLAB / Octave

```matlab
cd matlab
dress_build   % compiles the MEX file
```
