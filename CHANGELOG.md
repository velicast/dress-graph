# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.8.3] - 2026-05-04

### Added
- **C++ `DRESS::loadFitResult(edge_dress, vertex_dress)`** (`libdress++/include/dress/dress.hpp`): copies externally-computed per-edge and per-vertex DRESS values into the underlying `dress_graph_t`, with length checks against `E` and `N`. After calling it, all read accessors (`get()`, `edgeDress()`, `vertexDress()`, `edgeDressValues()`, `vertexDressValues()`) reflect the loaded values. Intended as the interop hook for hardware backends (CUDA/MPI) that compute results out-of-band.
- **Python `DRESS.load_fit_result(edge_dress, vertex_dress)`** (pybind11 binding): exposes the new C++ method, accepting any array-like coerced to C-contiguous `float64` arrays of shape `(E,)` and `(N,)`.

### Changed
- **Python `_sync_hardware_fit()`**: when the active backend is the C/pybind11 impl and `load_fit_result` is available, hardware-backend results (from `fit_cuda`, `fit_mpi`, etc.) are now copied directly into the C graph struct instead of being mirrored into Python lists. This keeps the C-backed `get()` / `edge_dress()` / `vertex_dress()` paths consistent with the hardware-computed state in the same process.
- **Python `DRESS.fit_cuda()`**: now always syncs the hardware fit result back into the local impl (previously the sync was skipped whenever the C backend was active, leaving the C graph holding stale values after a CUDA run).
- **Python `dress.cuda.DRESS`**: removed the `_force_python_impl = True` override. CPU-side queries on a CUDA graph (e.g. `g.get(u, v)` after `g.fit()`) now run through the C backend using the synced converged state, instead of forcing a slower pure-Python fallback.

### Fixed
- **Python `_DressGraph` ctypes layout** (`python/src/dress/_ctypes_helpers.py`): reordered the `NW` and `vertex_dress` fields to match the actual `dress_graph_t` C struct layout. The previous order misaligned every field after `edge_dress_next`, which could corrupt memory or return wrong values when accessing vertex weights / vertex DRESS through the ctypes path.
- **Python `_sync_hardware_fit()` attribute names**: replaced leftover `_node_dress` / `_np_node_dress` references (missed during the v0.8.0 node→vertex rename) with `_vertex_dress` / `_np_vertex_dress`. Without this fix, the pure-Python fallback path silently failed to update vertex DRESS values and to invalidate the cached NumPy view after a hardware fit.

## [0.8.2] - 2026-04-09

### Fixed
- **Python `dress.omp` and `dress.mpi` auto-build**: these backends loaded `libdress.so` via `ctypes.CDLL()` but the pip wheel did not bundle the shared library, causing `RuntimeError: libdress.so not found` on first use. Both modules now auto-compile `libdress.so` from the vendored C sources on first call (same pattern as `dress.cuda`), requiring only a system C compiler (`gcc` for OMP, `mpicc` for MPI). No `LD_LIBRARY_PATH` or manual build step needed.
- **Python `dress.omp` / `dress.mpi` search path**: added the package's own directory to the `ctypes.CDLL` search candidates, so manually placed or symlinked `.so` files are found.

## [0.8.1] - 2026-04-03

### Fixed
- **Python vendored CUDA backends**: fixed the auto-build source lists in `dress.cuda` and `dress.mpi.cuda` so the generated shared libraries include the nabla implementation sources (`nabla_dress.c`, `nabla_dress_impl.c`, `nabla_dress_cuda.c`) alongside the existing delta sources. This fixes runtime symbol mismatches where the Python wrappers bound `dress_nabla_fit_*` functions that were missing from the rebuilt `.so`

## [0.8.0] - 2026-03-31

### Changed
- **Terminology: node → vertex** (BREAKING): renamed all "node" references to "vertex" across the entire codebase for consistency with graph-theoretic convention. Affects all 9 language bindings, the C struct, C++ methods, docs, tests, and examples:
  - C struct: `node_dress` → `vertex_dress`, `node_weights` → `vertex_weights` (in `dress_graph_t`)
  - C++: `nodeDress()` → `vertexDress()`, `nodeDressValues()` → `vertexDressValues()`
  - Python: `node_dress` → `vertex_dress`, `node_weights` → `vertex_weights`, `node_dress_values` → `vertex_dress_values` (result fields + DRESS class properties)
  - Rust: `node_dress` → `vertex_dress`, `node_weights` → `vertex_weights` (in `DressResult`)
  - Go: `NodeDress` → `VertexDress`, `NodeWeights` → `VertexWeights` (in `Result`)
  - Julia: `node_dress` → `vertex_dress`, `node_weights` → `vertex_weights` (in `DRESSResult`)
  - R: `node_dress` → `vertex_dress`, `node_weights` → `vertex_weights` (in result lists and function parameters)
  - MATLAB/Octave: `.node_dress` → `.vertex_dress` (in result structs)
  - WASM/JS: `nodeDress` → `vertexDress`, `nodeWeights` → `vertexWeights` (in result objects and options)
  - igraph: `node_weight_attr` → `vertex_weight_attr`
  - NetworkX: `dress_norm` attribute → `vertex_dress` attribute (written by `fit(G, set_attributes=True)`)
  - All docs, comments, and docstrings updated: "per-node" → "per-vertex"

### Fixed
- **Windows MSVC wheel build**: added portable `rand_r` shim (`dress_rand_r`) for `delta_dress_impl.c` and `nabla_dress_impl.c` — POSIX `rand_r()` does not exist on Windows, causing linker failure `LNK2001: unresolved external symbol rand_r`
- **CI wheel vendor step**: updated `build_wheels.yml` to include nabla sources (`nabla_dress.c`, `nabla_dress_impl.*`), histogram sources (`dress_histogram.*`), OMP sources, and removed deleted `delta_dress.h` — fixes wheel build failures on all three platforms
- **R docs**: removed duplicated `\alias{dress_version}` from `fit.Rd` (now only in `dress_version.Rd`); removed stale `offset`/`stride` parameters from `delta_fit.Rd`
- **Δᵏ-DRESS multiset NaN initialization**: the multisets matrix for `dress_delta_fit` is now initialized with `NaN` instead of zero (`calloc`), matching the `nabla` implementation. Previously, unsampled rows (when using `n_samples`) contained zeros, which were indistinguishable from legitimate edge values. Now unsampled rows are all-`NaN`, consistent with deleted-edge markers within sampled rows

## [0.7.0] - 2026-03-26

### Added
- **∇ᵏ-DRESS (Nabla)**: vertex individualization lifting operator across the entire codebase. Enumerates all P(N,k) ordered k-tuples, marks each with distinct generic weights (√primes), runs DRESS on each marked graph, and accumulates converged edge values into a sparse exact histogram. Available across all backends and all 9 language bindings:
  - C: `dress_nabla_fit()`, `dress_nabla_fit_flat()`, `dress_nabla_fit_strided_flat()`
  - C (OMP): `dress_nabla_fit_omp()`, `dress_nabla_fit_omp_strided()`
  - C (CUDA): `dress_nabla_fit_cuda()`, `dress_nabla_fit_cuda_strided()`
  - C (MPI): `dress_nabla_fit_mpi()`, `dress_nabla_fit_mpi_cuda()`, `dress_nabla_fit_mpi_omp()`
  - C++: `DRESS::nablaFit()` method on all backend classes
  - Python: `nabla_fit()` free function + `DRESS.nabla_fit()` method + NetworkX `nabla_fit()`
  - Rust: `nabla_fit()` free function + `DRESS::nabla_fit()` method
  - Go: `NablaFit()` free function + `DRESS.NablaFit()` method
  - Julia: `nabla_fit()` free function + `nabla_fit!()` method on `DressGraph`
  - R: `nabla_fit()` + `omp$nabla_fit()` + `cuda$nabla_fit()` + `mpi$nabla_fit()` + `mpi$cuda$nabla_fit()` + `mpi$omp$nabla_fit()`
  - MATLAB/Octave: `nabla_fit()` + `nabla_dress_mex` / `nabla_dress_omp_mex` / `nabla_dress_cuda_mex` MEX entry points
  - WASM: `nablaFit()` async function + `DRESS.nablaFit()` method
- **Node weights** across the core API and language bindings: DRESS constructors and free-function wrappers now accept per-vertex weights while preserving the existing implicit all-ones default when node weights are omitted
- **Sampled Δᵏ-DRESS** (`n_samples`, `seed`): all `dress_delta_fit` variants now accept optional random sampling parameters, placed immediately after `epsilon` in every API. When `n_samples > 0`, only a random subset of the C(N,k) deletion subsets is evaluated (DFS pick-filter with sorted dedup). Sampling is available across all backends (CPU, OMP, CUDA, MPI, MPI+OMP, MPI+CUDA) and all language bindings (C, C++, Python, Rust, Go, Julia, R, MATLAB/Octave, WASM, igraph). Defaults: `n_samples=0` (exhaustive), `seed=0`
- **Optional histogram** (`compute_histogram`): all `dress_delta_fit` variants accept a flag to skip histogram construction when only multisets are needed. In the C API, passing `NULL` for `hist_size` skips all histogram work. Exposed as `compute_histogram` (default `true`) in C++, Python, Rust, Go, Julia, R, MATLAB/Octave, WASM, and igraph bindings
- **igraph multisets**: `delta_dress_result_igraph_t` now includes `multisets` and `num_subgraphs` fields; `dress_delta_fit_igraph` and all MPI igraph variants accept `keep_multisets` and `compute_histogram` parameters
- **OpenMP backend** (`dress/omp/`): new `dress_fit_omp` (edge-parallel) and `dress_delta_fit_omp` (subgraph-parallel) functions separate OpenMP parallelism from the sequential core. Available in all language bindings:
  - C: `#include "dress/omp/dress.h"` — drop-in redirect header
  - C++: `omp::DRESS` class in `dress/omp/dress.hpp`
  - Python: `from dress.omp import DRESS` (+ NetworkX: `dress.omp.networkx`)
  - Rust: `dress_graph::omp` module (feature `omp`)
  - Go: `github.com/velicast/dress-graph/go/omp` package
  - Julia: `using DRESS.OMP`
  - R: `omp$fit()` / `omp$delta_fit()` / `omp$nabla_fit()`
  - MATLAB/Octave: `omp.fit()` / `omp.delta_fit()` / `omp.nabla_fit()`
- **MPI+OpenMP backend** (`dress/mpi/omp/`): new `dress_delta_fit_mpi_omp` and `dress_nabla_fit_mpi_omp` distribute subgraphs/tuples across MPI ranks with OpenMP intra-rank parallelism. Available in all language bindings
- **Examples** for OMP and MPI+OMP added for all 9 languages (30 new example files under `examples/`)

### Changed
- **C API naming convention** (BREAKING): all public C functions now use `dress_` as a consistent prefix. Renames:
  - `init_dress_graph` → `dress_init_graph`
  - `free_dress_graph` → `dress_free_graph`
  - `delta_dress_fit*` → `dress_delta_fit*`
  - `nabla_dress_fit*` → `dress_nabla_fit*`
  - `delta_binom_public` → `dress_delta_binom`
  - `nabla_perm_count` → `dress_nabla_perm_count`
  - Removed backward-compatibility macros (`fit()`, `delta_fit()`) and the `delta_dress.h` shim header
  - All upstream wrappers (C++, Python ctypes, Rust FFI, Go CGo, Julia dlsym, R .Call, MATLAB MEX, WASM exports) updated to use new names
- **Wrapper function naming** (BREAKING): removed redundant `dress_` prefix from scoped/namespaced standalone functions across all high-level language bindings:
  - Python: `dress_fit` → `fit`, `delta_dress_fit` → `delta_fit`, `nabla_dress_fit` → `nabla_fit` (+ NetworkX: `dress_graph` → `fit`, `delta_dress_graph` → `delta_fit`)
  - Rust: `pub fn dress_fit` → `pub fn fit`, `pub fn delta_dress_fit` → `pub fn delta_fit`, `pub fn nabla_dress_fit` → `pub fn nabla_fit`
  - Go: `DressFit` → `Fit`, `DeltaDressFit` → `DeltaFit`, `NablaDressFit` → `NablaFit`
  - Julia: `dress_fit` → `fit`, `delta_dress_fit` → `delta_fit`, `nabla_dress_fit` → `nabla_fit`
  - R: `dress_fit` → `fit`, `delta_dress_fit` → `delta_fit`, `nabla_dress_fit` → `nabla_fit` (+ environment members: `omp$dress_fit` → `omp$fit`, etc.)
  - MATLAB/Octave: files renamed (`dress_fit.m` → `fit.m`, `delta_dress_fit.m` → `delta_fit.m`, `nabla_dress_fit.m` → `nabla_fit.m`) across all backends
  - WASM/JS: `dressFit` → `fit`, `deltaDressFit` → `deltaFit`, `nablaDressFit` → `nablaFit`
- **Rust API redesign**: removed builder pattern (`DRESSBuilder`, `.build()`, `.build_and_fit()`) across all backends. New API uses `DRESS::new()`, `g.delta_fit()`, `g.nabla_fit()`, and module-level `fit()` / `delta_fit()` / `nabla_fit()` free functions
- **C++ namespace**: all classes wrapped in `dress::` namespace (`dress::DRESS`, `dress::cuda::DRESS`, etc.)
- **Removed offset/stride** from all public-facing APIs (Python, Rust, Go, Julia, R, MATLAB, WASM). Strided computation remains available in the internal C API for MPI distribution
- **Sequential core** (`dress/dress.h`): `dress_fit` is now purely sequential — no OpenMP pragmas
- **Histogram representation** for Δᵏ-DRESS and ∇ᵏ-DRESS results is now a sparse exact histogram of sorted `(value, count)` entries instead of dense epsilon-binned arrays
- **R MPI `delta_fit` wrappers**: added missing `n_samples`, `seed`, `compute_histogram` parameters to `mpi$delta_fit`, `mpi$cuda$delta_fit`, and `mpi$omp$delta_fit` — fixes argument mismatch with the C glue layer

### Fixed
- **Octave MPI OO support**: `mpi.DRESS(...).delta_fit(...)` and `mpi.cuda.DRESS(...).delta_fit(...)` now work in packaged Octave builds instead of shipping placeholder wrappers that errored at runtime
- **Octave package build/runtime parity**: the Octave package now vendors and builds the required MPI and MPI+CUDA MEX sources, and `run_examples.sh` rebuilds a fresh tarball and checks backend availability before running MPI examples
- **Octave MPI+CUDA linking**: the MPI+CUDA object-delta MEX target now links the required core delta implementation, fixing the missing-symbol failure around `dress_delta_fit_strided_flat`
- **R OpenMP flags**: `r/configure` now always passes `-fopenmp` to `PKG_CFLAGS`/`PKG_LIBS`, ensuring `_OPENMP` is defined for all source files (fixes MPI+OMP symbol registration on systems where `${SHLIB_OPENMP_CFLAGS}` is empty)

## [0.6.2] - 2026-03-22

### Fixed
- **CUDA wheel packaging**: CI vendoring step now copies CUDA and MPI sources (`dress_cuda.cu`, `dress_cuda.h`, `delta_dress_cuda.c`, `dress_mpi.c`) into wheels, matching `publish.sh` — users with `nvcc` can auto-build the CUDA backend from PyPI installs
- **CUDA stale-check false positive**: `_sources_newer_than()` in `cuda/__init__.py` and `mpi/cuda/__init__.py` now skips the stale check when `dress_cuda.cu` is absent (pre-built wheel), preventing a spurious rebuild attempt that failed because the `.cu` source wasn't shipped

## [0.6.1] - 2026-03-16

### Fixed
- **CUDA workspace sizing**: the CUDA DRESS backend no longer allocates temporary scratch space for every node and edge from the graph-wide `max_degree`. It now plans node and edge work in batches using exact per-item scratch requirements and reuses a shared device workspace, which eliminates out-of-memory failures on large skewed graphs while preserving exact bitwise-equal CPU/CUDA results.

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
  - C: `#include "dress/mpi/dress.h"` redirects `dress_delta_fit()` → `dress_delta_fit_mpi(..., MPI_COMM_WORLD)`; explicit `dress_delta_fit_mpi()` / `dress_delta_fit_mpi_fcomm()` available for custom communicators
  - C: `#include "dress/mpi/cuda/dress.h"`: single convenience header for MPI + CUDA (GPU-distributed)
  - C++: `mpi::DRESS`, `mpi::cuda::DRESS` namespaces
  - Python: `from dress.mpi import dress_delta_fit` (CPU), `from dress.mpi.cuda import dress_delta_fit` (GPU)
  - Python NetworkX: `from dress.mpi.networkx import delta_dress_graph` / `from dress.mpi.cuda.networkx import delta_dress_graph`
  - R: `mpi$dress_delta_fit()`, `mpi$cuda$dress_delta_fit()`
  - Rust: `dress_graph::mpi::delta_fit()`, `dress_graph::mpi::cuda::delta_fit()`
  - Go: `go/mpi`, `go/mpi/cuda` import paths
  - Julia: `DRESS.MPI`, `DRESS.MPI.CUDA` modules
- **igraph wrapper restructured** (`libdress-igraph`):
  - Headers moved to `dress/igraph/dress.h`, `dress/cuda/igraph/dress.h`, `dress/mpi/igraph/dress.h`, `dress/mpi/cuda/igraph/dress.h`
  - **Convenience macros**: `dress_fit`, `dress_free`, `dress_to_vector`, `dress_delta_fit`, `delta_dress_free`, `delta_dress_to_vector` map to their `_igraph` counterparts; user code uses the same names as the core API
  - MPI support: `dress/mpi/igraph/dress.h` redirects `dress_delta_fit()` to MPI backend (`dress_delta_fit_mpi_igraph` / `dress_delta_fit_mpi_cuda_igraph` + `_fcomm` FFI variants)
  - `dress/mpi/cuda/igraph/dress.h`: single header for MPI + CUDA igraph
  - `dress_igraph_mpi.c` implementation using shared `delta_dress_impl.h`
  - CMake targets: `dress_igraph_mpi_static`, `dress_igraph_mpi_shared`
  - Examples link against `libdress.a` instead of recompiling core sources
- Octave CUDA support: `cuda.dress_fit()`, `cuda.dress_delta_fit()` via `+cuda` namespace

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
- CUDA support for igraph wrapper via `#include <dress/cuda/igraph/dress.h>`: redirects `dress_fit()` / `dress_delta_fit()` to CUDA via macros
- CUDA support via import-based switching across all language wrappers
  - C: `#include "dress/cuda/dress.h"` redirects `dress_fit()` / `dress_delta_fit()` to CUDA via macros (also available as explicit `dress_fit_cuda()` in `dress/cuda/dress_cuda.h`)
  - C++: `dress::cuda::fit()`, `dress::cuda::delta_fit()`, same names as CPU, different namespace
  - Python: `from dress.cuda import dress_fit, dress_delta_fit`
  - R: `dress.graph::cuda$dress_fit()`, `dress.graph::cuda$dress_delta_fit()`
  - Rust: `dress_graph::cuda::dress_fit()`, `dress_graph::cuda::dress_delta_fit()`
  - Go: `dress.DressFit()` / `dress.DeltaDressFit()`. Switch by changing import path (`go/` → `go/cuda/`); old names `Fit`/`DeltaFit` kept as deprecated aliases
  - Julia: `DRESS.CUDA.dress_fit()`, `DRESS.CUDA.dress_delta_fit()`
  - MATLAB/Octave: `cuda.dress_fit()`, `cuda.dress_delta_fit()`, same names as CPU, different package (`+cuda`)
  - WASM: `dressFit()`, `deltaDressFit()` (CPU only, no CUDA in browser)
- `dress.cuda.networkx` module: GPU-accelerated NetworkX helpers
- `delta_dress_impl.h`: shared internal implementation for Δ^k-DRESS, parameterized by fit function pointer (eliminates code duplication between CPU and CUDA)
- R: `cuda.Rd` man page documenting the CUDA environment
- Julia: `DRESSResult` added to module exports

### Changed
- **igraph wrapper renamed**: `dress_igraph_compute` → `dress_fit_igraph`, `dress_igraph_free` → `dress_free_igraph`, `dress_igraph_to_vector` → `dress_to_vector_igraph`, `dress_igraph_result_t` → `dress_result_igraph_t` (and corresponding `delta_` variants)
- Python NetworkX API: GPU-accelerated helpers moved to dedicated `dress.cuda.networkx` module
- Python: `dress.cuda` converted from single file to package (`dress/cuda/`)
- `delta_dress.c` and `delta_dress_cuda.c` refactored to thin wrappers delegating to `dress_delta_fit_impl()`
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
