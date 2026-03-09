# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.5.0] — 2026-03-09

### Added
- **MPI support** — distributed Δᵏ-DRESS across all language bindings via import/include-based switching:
  - C: `#include "dress/mpi/dress.h"` redirects `delta_dress_fit()` → `delta_dress_fit_mpi(..., MPI_COMM_WORLD)`; explicit `delta_dress_fit_mpi()` / `delta_dress_fit_mpi_fcomm()` available for custom communicators
  - C: `#include "dress/mpi/cuda/dress.h"` — single convenience header for MPI + CUDA (GPU-distributed)
  - C++: `mpi::DRESS`, `mpi::cuda::DRESS` namespaces
  - Python: `from dress.mpi import delta_dress_fit` (CPU), `from dress.mpi.cuda import delta_dress_fit` (GPU)
  - Python NetworkX: `from dress.mpi.networkx import delta_dress_graph` / `from dress.mpi.cuda.networkx import delta_dress_graph`
  - R: `mpi$delta_dress_fit()`, `mpi$cuda$delta_dress_fit()`
  - Rust: `dress_graph::mpi::delta_fit()`, `dress_graph::mpi::cuda::delta_fit()`
  - Go: `go/mpi`, `go/mpi/cuda` import paths
  - Julia: `DRESS.MPI`, `DRESS.MPI.CUDA` modules
- **igraph wrapper restructured** (`libdress-igraph`):
  - Headers moved to `dress/igraph/dress.h`, `dress/cuda/igraph/dress.h`, `dress/mpi/igraph/dress.h`, `dress/mpi/cuda/igraph/dress.h`
  - **Convenience macros** — `dress_fit`, `dress_free`, `dress_to_vector`, `delta_dress_fit`, `delta_dress_free`, `delta_dress_to_vector` map to their `_igraph` counterparts; user code uses the same names as the core API
  - MPI support: `dress/mpi/igraph/dress.h` redirects `delta_dress_fit()` to MPI backend (`delta_dress_fit_mpi_igraph` / `delta_dress_fit_mpi_cuda_igraph` + `_fcomm` FFI variants)
  - `dress/mpi/cuda/igraph/dress.h` — single header for MPI + CUDA igraph
  - `dress_igraph_mpi.c` implementation using shared `delta_dress_impl.h`
  - CMake targets: `dress_igraph_mpi_static`, `dress_igraph_mpi_shared`
  - Examples link against `libdress.a` instead of recompiling core sources
- Octave CUDA support: `cuda.dress_fit()`, `cuda.delta_dress_fit()` via `+cuda` namespace

### Fixed
- Static CUDA linking across all 9 language bindings (37/37 examples pass)
- R build for CRAN compatibility
- `set_version.sh` now updates the Octave tarball version in `run_examples.sh`

### Changed
- `clean.sh` updated to clean all new igraph example binaries
- README and docs updated: igraph sections in API reference, `mpi/cuda/dress.h` convenience header documented, backend support matrix noted in language bindings section
- **API reference rewritten** — README and `docs/getting-started/api.md` now document the full C backend matrix (CPU, CUDA, MPI, MPI+CUDA) with correct header names and link flags, plus igraph wrapper rows for all four backends
- **Examples collection** — 44 end-to-end examples across 9 languages (C, C++, Python, Rust, Go, Julia, R, Octave, WASM), covering CPU, CUDA, MPI, and MPI+CUDA backends; verified via `run_examples.sh`

## [0.4.0] — 2026-03-08

### Added
- CUDA support for igraph wrapper via `#include <dress/cuda/igraph/dress.h>` — redirects `dress_fit()` / `delta_dress_fit()` to CUDA via macros
- CUDA support via import-based switching across all language wrappers
  - C: `#include "dress/cuda/dress.h"` redirects `dress_fit()` / `delta_dress_fit()` to CUDA via macros (also available as explicit `dress_fit_cuda()` in `dress/cuda/dress_cuda.h`)
  - C++: `dress::cuda::fit()`, `dress::cuda::delta_fit()` — same names as CPU, different namespace
  - Python: `from dress.cuda import dress_fit, delta_dress_fit`
  - R: `dress.graph::cuda$dress_fit()`, `dress.graph::cuda$delta_dress_fit()`
  - Rust: `dress_graph::cuda::dress_fit()`, `dress_graph::cuda::delta_dress_fit()`
  - Go: `dress.DressFit()` / `dress.DeltaDressFit()` — switch by changing import path (`go/` → `go/cuda/`); old names `Fit`/`DeltaFit` kept as deprecated aliases
  - Julia: `DRESS.CUDA.dress_fit()`, `DRESS.CUDA.delta_dress_fit()`
  - MATLAB/Octave: `cuda.dress_fit()`, `cuda.delta_dress_fit()` — same names as CPU, different package (`+cuda`)
  - WASM: `dressFit()`, `deltaDressFit()` (CPU only — no CUDA in browser)
- `dress.cuda.networkx` module — GPU-accelerated NetworkX helpers
- `delta_dress_impl.h` — shared internal implementation for Δ^k-DRESS, parameterized by fit function pointer (eliminates code duplication between CPU and CUDA)
- R: `cuda.Rd` man page documenting the CUDA environment
- Julia: `DRESSResult` added to module exports

### Changed
- **igraph wrapper renamed**: `dress_igraph_compute` → `dress_fit_igraph`, `dress_igraph_free` → `dress_free_igraph`, `dress_igraph_to_vector` → `dress_to_vector_igraph`, `dress_igraph_result_t` → `dress_result_igraph_t` (and corresponding `delta_` variants)
- Python NetworkX API: GPU-accelerated helpers moved to dedicated `dress.cuda.networkx` module
- Python: `dress.cuda` converted from single file to package (`dress/cuda/`)
- `delta_dress.c` and `delta_dress_cuda.c` refactored to thin wrappers delegating to `delta_dress_fit_impl()`
- Vendoring (`build.sh`, `setup.py`, `build.rs`) updated to include `delta_dress_impl.h`

### Fixed
- R: missing `export(cuda)` in NAMESPACE
- R: `.Call` symbol resolution crash on non-CUDA builds (switched to runtime string-based lookup)

### Added
- C API: `dress_get()` — query the DRESS similarity for any edge (existing or virtual) on a fitted graph without rebuilding (`double dress_get(p_dress_graph_t, u, v, max_iterations, epsilon, edge_weight)`)
- Persistent `DRESS` objects across all language wrappers — hold the C graph alive for repeated `fit` / `get` (virtual edge query) calls without rebuilding:
  - Go: `DRESS` struct with `NewDRESS()` → `Fit()` → `Get()` → `Result()` → `Close()`; CUDA variant uses `dress_fit_cuda` for `Fit()`
  - Rust: `DRESS` with RAII (`Drop`) — `DRESS::builder().build()` → `fit()` → `get()` → `result()` → `close()`; CUDA variant in `dress_graph::cuda::DRESS`
  - R: environment-based `DRESS()` class with GC finalizer (`$fit()`, `$get()`, `$result()`, `$close()`); backed by `EXTPTRSXP` + `R_RegisterCFinalizerEx`
  - Julia: mutable `DressGraph` struct with GC finalizer — `DressGraph()` → `fit!()` → `get()` → `result()` → `close!()`
  - MATLAB/Octave: `DRESS` handle class + five new MEX gateways (`dress_init_mex`, `dress_fit_obj_mex`, `dress_get_mex`, `dress_result_mex`, `dress_free_mex`)
  - WASM/JS: `DressGraph` class with async `create()` factory — `.fit()` → `.get()` → `.result()` → `.free()`; TypeScript declarations included
  - Python/NetworkX: `NxDRESS` persistent class in `dress.networkx` — wraps NetworkX graphs with node-label translation, supports `fit()` → `get(u, v)` → `result()` → `close()` and context-manager protocol
- `dress_get` re-exported from WASM build (`EXPORTED_FUNCTIONS`)
- Rust FFI: `dress_get` declaration restored in both CPU and CUDA extern blocks
- Julia: `_FN_GET` function pointer for `dress_get`; `fit!`, `close!` added to module exports
- Octave: updated `INDEX`, `Makefile`, and tarball build to include new MEX files and `DRESS.m`
