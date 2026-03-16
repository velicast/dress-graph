# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.6.0] - 2026-03-16

### Added
- **Sort + KBN compensated summation** in C, CUDA and Pure-Python backends — ensures DRESS fingerprints are true graph invariants independent of vertex labeling (bitwise-equal results regardless of accumulation order):
  - `dress.c`: all floating-point accumulations (node norms, edge dress, virtual-edge intercepts) now use sorted Kahan–Babuška–Neumaier (KBN) summation with stack-allocated buffers (heap fallback above 4096 terms)
  - `dress_cuda.cu`: GPU kernels use per-thread iterative quicksort (median-of-three pivot, insertion-sort cutoff at n ≤ 16, 64-entry stack) followed by KBN summation on global-memory workspace
- **`max_degree` field** in `dress_graph_t` struct: computed O(1) inline during CSR construction; used by the CUDA backend to size per-thread workspace dynamically
- **CUDA Makefile** (`libdress/src/cuda/Makefile`): standalone build for `libdress_cuda.so` (nvcc + gcc/mpicc), compiles `dress_cuda.cu`, `delta_dress_cuda.c`, `delta_dress_impl.c`, and optionally `dress_mpi.c`
- **Tests**: 346 new lines in `tests/c/test_dress.c` (sort+KBN correctness, compensated-sum edge cases) and 155 new lines in `tests/python/test_dress_core.py` (Python sort+KBN parity tests)
- `publish.sh`: vendors `dress_mpi.c` into PyPI wheel (matching Rust/R/Go) so `dress.mpi.cuda` auto-build works from vendored sources

### Fixed
- **Python CUDA auto-build**: `cuda/__init__.py` and `mpi/cuda/__init__.py` now use separate `.so` filenames (`libdress_cuda.so` vs `libdress_mpi_cuda.so`) preventing the MPI+CUDA build from clobbering the plain CUDA `.so` — and vice versa
- **Python CUDA vendoring**: fixed `_libdress` → `_vendored` path in `cuda/__init__.py`; added vendored-path fallback in `mpi/cuda/__init__.py`
- **Python auto-build staleness**: both CUDA modules detect when vendored sources are newer than the cached `.so` and rebuild automatically
- **Python install isolation**: `publish.sh --install-local` now uninstalls any editable install first (removes `.pth` redirect), installs the wheel with `--no-deps`, and cleans stale `.so`/`.o` from the vendored directory
- **Venv discovery**: `run_examples.sh` and `publish.sh` search `$ROOT/.venv` then `$ROOT/../.venv`, supporting both project-local and parent-directory venv layouts
- `clean.sh`: no longer deletes `libdress/src/cuda/Makefile` (it's a source file)
- `.gitignore`: added `!libdress/src/cuda/Makefile` and `!octave/src/Makefile` exceptions to the blanket `Makefile` rule
- Rust: removed unused `use mpi::ffi::MPI_Comm` import in CUDA submodule

### Changed
- **Struct offsets updated** across all FFI bindings (Rust, Go, Julia, WASM, Python) to account for the new `max_degree` field at offset 56 (LP64) / 36 (wasm32)
- `run_examples.sh`: 63 examples pass across 9 languages (C, C++, Python, Rust, Go, Julia, R, Octave, WASM), 0 failures

## [0.5.3] - 2026-03-15

### Changed
- **WASM/JS**: renamed OO class `DressGraph` → `DRESS` and options type `DressGraphOptions` → `DRESSOptions` for consistency with other language wrappers

## [0.5.2] - 2026-03-15

### Added
- **OO API examples**: 26 new end-to-end examples demonstrating the persistent `DRESS` object (construct → fit → get → result → close) across all language wrappers:
  - Python: `cpu_oo.py`, `cuda_oo.py`, `mpi_oo.py`, `mpi_cuda_oo.py`
  - Rust: `cpu_oo.rs`, `cuda_oo.rs`, `mpi_oo.rs`, `mpi_cuda_oo.rs`
  - Go: `cpu_oo.go`, `cuda_oo.go`, `mpi_oo.go`, `mpi_cuda_oo.go`
  - Julia: `cpu_oo.jl`, `cuda_oo.jl`, `mpi_oo.jl`, `mpi_cuda_oo.jl`
  - R: `cpu_oo.R`, `cuda_oo.R`, `mpi_oo.R`, `mpi_cuda_oo.R`
  - MATLAB: `cpu_oo.m`, `cuda_oo.m`, `mpi_oo.m`, `mpi_cuda_oo.m`
  - Octave: `cpu_oo.m`
  - WASM: `cpu_oo.mjs`
- `publish.sh pypi`: PyPI publishing via `gh release create` triggering `build_wheels.yml` GitHub Actions workflow (cibuildwheel cross-platform builds + trusted publishing)
- MATLAB section in `run_examples.sh` (skipped when MATLAB unavailable)
- Octave `src/Makefile` for MEX compilation

### Fixed
- `build.sh`: missing `+cuda/DRESS.m`, `+mpi/DRESS.m`, `+mpi/+cuda/DRESS.m`, and `dress_fit_cuda_obj_mex.c` in Octave vendor step
- `clean.sh`: added cleanup for CUDA build artifacts, `rust/LICENSE`, Python egg-info variants, `wasm/.npmrc`, Go vendor dirs, Julia `vendor/libdress.so`, Octave `inst/` and `src/` build outputs, R vendored sources, LaTeX auxiliary files; preserved `octave/src/Makefile`
- Octave `INDEX`: added CUDA, MPI, and MPI+CUDA sections
- Rust `examples/Cargo.toml`: registered OO example binaries
- Julia OO examples: fixed imports (`using DRESS` / `using DRESS.CUDA`)
- Rust MPI OO examples: fixed `Option<Vec<f64>>` unwrap for multisets

### Changed
- README: API reference tables updated with OO example links for all languages
- `run_examples.sh`: all 26 OO examples registered (63 pass, 0 fail, 1 skip)

## [0.5.1] - 2026-03-10

### Fixed
- npm package (`dress-graph`): `files` array in `wasm/package.json` listed `dress_wasm.js` but the Emscripten build produces `dress_wasm.cjs`. Anyone installing 0.5.0 from npm got `Cannot find module './dress_wasm.cjs'`
- `run_examples.sh`: `run_wasm()` now calls `publish.sh --install-local npm` before running examples, matching the install-then-test pattern used by Python, Rust, Julia, R, and Octave
- `clean.sh`: added cleanup of `examples/wasm/node_modules/`, `package-lock.json`, and `package.json`

## [0.5.0] - 2026-03-09

### Added
- **MPI support**: distributed Δᵏ-DRESS across all language bindings via import/include-based switching:
  - C: `#include "dress/mpi/dress.h"` redirects `delta_dress_fit()` → `delta_dress_fit_mpi(..., MPI_COMM_WORLD)`; explicit `delta_dress_fit_mpi()` / `delta_dress_fit_mpi_fcomm()` available for custom communicators
  - C: `#include "dress/mpi/cuda/dress.h"`: single convenience header for MPI + CUDA (GPU-distributed)
  - C++: `mpi::DRESS`, `mpi::cuda::DRESS` namespaces
  - Python: `from dress.mpi import delta_dress_fit` (CPU), `from dress.mpi.cuda import delta_dress_fit` (GPU)
  - Python NetworkX: `from dress.mpi.networkx import delta_dress_graph` / `from dress.mpi.cuda.networkx import delta_dress_graph`
  - R: `mpi$delta_dress_fit()`, `mpi$cuda$delta_dress_fit()`
  - Rust: `dress_graph::mpi::delta_fit()`, `dress_graph::mpi::cuda::delta_fit()`
  - Go: `go/mpi`, `go/mpi/cuda` import paths
  - Julia: `DRESS.MPI`, `DRESS.MPI.CUDA` modules
- **igraph wrapper restructured** (`libdress-igraph`):
  - Headers moved to `dress/igraph/dress.h`, `dress/cuda/igraph/dress.h`, `dress/mpi/igraph/dress.h`, `dress/mpi/cuda/igraph/dress.h`
  - **Convenience macros**: `dress_fit`, `dress_free`, `dress_to_vector`, `delta_dress_fit`, `delta_dress_free`, `delta_dress_to_vector` map to their `_igraph` counterparts; user code uses the same names as the core API
  - MPI support: `dress/mpi/igraph/dress.h` redirects `delta_dress_fit()` to MPI backend (`delta_dress_fit_mpi_igraph` / `delta_dress_fit_mpi_cuda_igraph` + `_fcomm` FFI variants)
  - `dress/mpi/cuda/igraph/dress.h`: single header for MPI + CUDA igraph
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
- **API reference rewritten**: README and `docs/getting-started/api.md` now document the full C backend matrix (CPU, CUDA, MPI, MPI+CUDA) with correct header names and link flags, plus igraph wrapper rows for all four backends
- **Examples collection**: 44 end-to-end examples across 9 languages (C, C++, Python, Rust, Go, Julia, R, Octave, WASM), covering CPU, CUDA, MPI, and MPI+CUDA backends; verified via `run_examples.sh`

## [0.4.0] - 2026-03-08

### Added
- CUDA support for igraph wrapper via `#include <dress/cuda/igraph/dress.h>`: redirects `dress_fit()` / `delta_dress_fit()` to CUDA via macros
- CUDA support via import-based switching across all language wrappers
  - C: `#include "dress/cuda/dress.h"` redirects `dress_fit()` / `delta_dress_fit()` to CUDA via macros (also available as explicit `dress_fit_cuda()` in `dress/cuda/dress_cuda.h`)
  - C++: `dress::cuda::fit()`, `dress::cuda::delta_fit()`, same names as CPU, different namespace
  - Python: `from dress.cuda import dress_fit, delta_dress_fit`
  - R: `dress.graph::cuda$dress_fit()`, `dress.graph::cuda$delta_dress_fit()`
  - Rust: `dress_graph::cuda::dress_fit()`, `dress_graph::cuda::delta_dress_fit()`
  - Go: `dress.DressFit()` / `dress.DeltaDressFit()`. Switch by changing import path (`go/` → `go/cuda/`); old names `Fit`/`DeltaFit` kept as deprecated aliases
  - Julia: `DRESS.CUDA.dress_fit()`, `DRESS.CUDA.delta_dress_fit()`
  - MATLAB/Octave: `cuda.dress_fit()`, `cuda.delta_dress_fit()`, same names as CPU, different package (`+cuda`)
  - WASM: `dressFit()`, `deltaDressFit()` (CPU only, no CUDA in browser)
- `dress.cuda.networkx` module: GPU-accelerated NetworkX helpers
- `delta_dress_impl.h`: shared internal implementation for Δ^k-DRESS, parameterized by fit function pointer (eliminates code duplication between CPU and CUDA)
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
- C API: `dress_get()`: query the DRESS similarity for any edge (existing or virtual) on a fitted graph without rebuilding (`double dress_get(p_dress_graph_t, u, v, max_iterations, epsilon, edge_weight)`)
- Persistent `DRESS` objects across all language wrappers: hold the C graph alive for repeated `fit` / `get` (virtual edge query) calls without rebuilding:
  - Go: `DRESS` struct with `NewDRESS()` → `Fit()` → `Get()` → `Result()` → `Close()`; CUDA variant uses `dress_fit_cuda` for `Fit()`
  - Rust: `DRESS` with RAII (`Drop`): `DRESS::builder().build()` → `fit()` → `get()` → `result()` → `close()`; CUDA variant in `dress_graph::cuda::DRESS`
  - R: environment-based `DRESS()` class with GC finalizer (`$fit()`, `$get()`, `$result()`, `$close()`); backed by `EXTPTRSXP` + `R_RegisterCFinalizerEx`
  - Julia: mutable `DressGraph` struct with GC finalizer: `DressGraph()` → `fit!()` → `get()` → `result()` → `close!()`
  - MATLAB/Octave: `DRESS` handle class + five new MEX gateways (`dress_init_mex`, `dress_fit_obj_mex`, `dress_get_mex`, `dress_result_mex`, `dress_free_mex`)
  - WASM/JS: `DRESS` class with async `create()` factory: `.fit()` → `.get()` → `.result()` → `.free()`; TypeScript declarations included
  - Python/NetworkX: `NxDRESS` persistent class in `dress.networkx`: wraps NetworkX graphs with node-label translation, supports `fit()` → `get(u, v)` → `result()` → `close()` and context-manager protocol
- `dress_get` re-exported from WASM build (`EXPORTED_FUNCTIONS`)
- Rust FFI: `dress_get` declaration restored in both CPU and CUDA extern blocks
- Julia: `_FN_GET` function pointer for `dress_get`; `fit!`, `close!` added to module exports
- Octave: updated `INDEX`, `Makefile`, and tarball build to include new MEX files and `DRESS.m`
