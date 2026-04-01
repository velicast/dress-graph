# API Reference

Every language binding exposes the same four operations (`fit`, `delta_fit`, `nabla_fit`, and `get`) plus a persistent graph object.
For usage examples see the [examples/](https://github.com/velicast/dress-graph/tree/main/examples) directory.

> **Backend switching:** In every language the API is identical; only the import or namespace changes.
> All backends: **CPU** (sequential), **OpenMP** (multi-threaded), **CUDA** (GPU), **MPI**, **MPI+OpenMP**, **MPI+CUDA**.
> WASM is CPU-only (browser).

---

## Common types

### Variants

| Constant | Value | Neighborhood \(N[u]\) | Combined weight \(\bar{w}(u,v)\) |
|----------|-------|------------------------|----------------------------------|
| `UNDIRECTED` | 0 | \(\{u\} \cup\) all neighbors (ignoring direction) | \(2\,w(u,v)\) |
| `DIRECTED` | 1 | \(\{u\} \cup\) all neighbors (in + out) | \(w(u,v) + w(v,u)\) |
| `FORWARD` | 2 | \(\{u\} \cup\) out-neighbors | \(w(u,v)\) |
| `BACKWARD` | 3 | \(\{u\} \cup\) in-neighbors | \(w(v,u)\) |

### `fit` result

| Field | Type | Description |
|-------|------|-------------|
| `sources` | int [E] | Edge source vertices (0-based) |
| `targets` | int [E] | Edge target vertices (0-based) |
| `edge_dress` | float [E] | DRESS similarity per edge |
| `edge_weight` | float [E] | Variant-specific edge weight |
| `vertex_dress` | float [N] | Per-vertex aggregated DRESS norm |
| `iterations` | int | Iterations performed |
| `delta` | float | Final maximum per-edge change |

### `delta_fit` result

| Field | Type | Description |
|-------|------|-------------|
| `histogram` | sparse entries | Exact histogram entries sorted by value ascending. Each entry stores a converged value and its multiplicity. |
| `hist_size` | int | Number of exact histogram entries. Exposed in low-level APIs; some high-level bindings omit it from the public result shape. |
| `multisets` | float [\(\binom{N}{k}\) × E] or null | Per-subgraph edge values (row-major; NaN = removed). Present when `keep_multisets` is enabled. |
| `num_subgraphs` | int64 | \(\binom{N}{k}\) : number of vertex-deletion subgraphs |

### `nabla_fit` result

| Field | Type | Description |
|-------|------|-------------|
| `histogram` | sparse entries | Exact histogram entries sorted by value ascending. Each entry stores a converged value and its multiplicity. |
| `hist_size` | int | Number of exact histogram entries. |
| `multisets` | float [\(P(N,k)\) × E] or null | Per-tuple edge values (row-major). Present when `keep_multisets` is enabled. |
| `num_tuples` | int64 | \(P(N,k)\) : number of ordered k-tuples |

### `dress_get` return

Single `float`: the DRESS value \(d_{uv}\) for one edge, computed in \(O(\deg u + \deg v)\) time on an already-fitted graph.

---

## C

Header: `dress/dress.h`

| Symbol | Signature |
|--------|-----------|
| `dress_variant_t` | `enum { DRESS_VARIANT_UNDIRECTED=0, DIRECTED=1, FORWARD=2, BACKWARD=3 }` |
| `dress_init_graph` | `p_dress_graph_t dress_init_graph(int N, int E, int *U, int *V, double *W, double *NW, dress_variant_t variant, int precompute_intercepts)` |
| `dress_fit` | `void dress_fit(p_dress_graph_t g, int max_iterations, double epsilon, int *iterations, double *delta)` |
| `dress_get` | `double dress_get(const p_dress_graph_t g, int u, int v, int max_iterations, double epsilon, double edge_weight)` |
| `dress_free_graph` | `void dress_free_graph(p_dress_graph_t g)` |

After fitting, result fields are accessed on the graph struct: `g->edge_dress[e]`, `g->edge_weight[e]`, `g->vertex_dress[u]`, `g->U[e]`, `g->V[e]`, `g->N`, `g->E`.

Header: `dress/delta_dress.h`

| Symbol | Signature |
|--------|-----------|
| `dress_hist_pair_t` | `struct { double value; int64_t count; }` |
| `dress_delta_fit` | `dress_hist_pair_t *dress_delta_fit(p_dress_graph_t g, int k, int iterations, double epsilon, int *hist_size, int keep_multisets, double **multisets, int64_t *num_subgraphs)` |
| `dress_nabla_fit` | `dress_hist_pair_t *dress_nabla_fit(p_dress_graph_t g, int k, int iterations, double epsilon, int n_samples, unsigned int seed, int *hist_size, int keep_multisets, double **multisets, int64_t *num_tuples)` |

Returns a `malloc`'d array of exact histogram pairs; caller must `free()`.

CUDA: include `dress/cuda/dress.h` instead; same signatures, redirected via macros.
MPI: include `dress/mpi/dress.h`; redirects `dress_delta_fit()` to MPI-distributed backend.
MPI+CUDA: include `dress/mpi/cuda/dress.h`: single header for GPU + distributed.

---

## C (igraph wrapper)

Header: `dress/igraph/dress.h`

> **Convenience macros:** The igraph header defines `dress_fit`, `dress_free`, `dress_to_vector`,
> `dress_delta_fit`, `delta_dress_free`, and `delta_dress_to_vector` as macros that expand to their
> `_igraph` counterparts. User code calls the same names as with the core API; the igraph backend
> is transparent.

| Type / Function | Description |
|--------|-----------|
| `dress_result_igraph_t` | Struct: `N`, `E`, `src[E]`, `dst[E]`, `dress[E]`, `weight[E]`, `vertex_dress[N]`, `iterations`, `delta` (zero-copy views) |
| `dress_fit` | `int dress_fit(const igraph_t *graph, const char *weight_attr, const char *vertex_weight_attr, dress_variant_t variant, int max_iters, double epsilon, int precompute, dress_result_igraph_t *result)` : macro for `dress_fit_igraph` |
| `dress_free` | `void dress_free(dress_result_igraph_t *result)` : macro for `dress_free_igraph` |
| `dress_to_vector` | `int dress_to_vector(const dress_result_igraph_t *result, igraph_vector_t *out)` : macro for `dress_to_vector_igraph` |
| `delta_dress_result_igraph_t` | Struct: `histogram[hist_size]` where each entry is `{value, count}`, `hist_size` |
| `dress_delta_fit` | `int dress_delta_fit(const igraph_t *graph, const char *weight_attr, const char *vertex_weight_attr, dress_variant_t variant, int k, int max_iters, double epsilon, int precompute, delta_dress_result_igraph_t *result)` : macro for `dress_delta_fit_igraph` |
| `delta_dress_free` | `void delta_dress_free(delta_dress_result_igraph_t *result)` : macro for `delta_dress_free_igraph` |
| `delta_dress_to_vector` | `int delta_dress_to_vector(const delta_dress_result_igraph_t *result, igraph_vector_t *out)` : macro for `delta_dress_to_vector_igraph`; flattens histogram entries as `[value0, count0, value1, count1, ...]` |

CUDA: include `dress/cuda/igraph/dress.h` instead; same calls, CUDA backend.
MPI: include `dress/mpi/igraph/dress.h` instead; redirects `dress_delta_fit()` to MPI. Adds:

| Function | Signature |
|--------|-----------|
| `dress_delta_fit_mpi_igraph` | `int dress_delta_fit_mpi_igraph(const igraph_t *graph, const char *weight_attr, const char *vertex_weight_attr, dress_variant_t variant, int k, int max_iters, double epsilon, int precompute, delta_dress_result_igraph_t *result, MPI_Comm comm)` |
| `dress_delta_fit_mpi_igraph_fcomm` | Same, but takes `int comm_f` (Fortran MPI handle) |
| `dress_delta_fit_mpi_cuda_igraph` | Same as above, CUDA+MPI backend (available when CUDA header is also included) |
| `dress_delta_fit_mpi_cuda_igraph_fcomm` | Fortran-handle variant of the CUDA+MPI function |

Uses the same `delta_dress_result_igraph_t`; free with `delta_dress_free()`.

MPI+CUDA: include `dress/mpi/cuda/igraph/dress.h` : single header that composes CUDA + MPI.

---

## C++

Header: `dress/dress.hpp`

**`DRESS` class**: RAII wrapper around the C graph.

| Member | Signature |
|--------|-----------|
| Constructor (unweighted) | `DRESS(int N, std::vector<int> sources, std::vector<int> targets, dress_variant_t variant, bool precomputeIntercepts)` |
| Constructor (weighted) | `DRESS(int N, std::vector<int> sources, std::vector<int> targets, std::vector<double> weights, dress_variant_t variant, bool precomputeIntercepts)` |
| `fit` | `FitResult fit(int maxIterations, double epsilon)` |
| `get` | `double get(int u, int v, int maxIterations = 100, double epsilon = 1e-6, double edgeWeight = 1.0) const` |
| `deltaFit` | `DeltaFitResult deltaFit(int k, int maxIterations, double epsilon, int nSamples = 0, unsigned int seed = 0, bool keepMultisets = false, bool computeHistogram = true)` |
| Accessors | `numVertices()`, `numEdges()`, `edgeDress(e)`, `edgeWeight(e)`, `vertexDress(u)`, `edgeSource(e)`, `edgeTarget(e)` |
| Bulk access | `edgeDressValues()`, `edgeWeights()`, `vertexDressValues()`, `edgeSources()`, `edgeTargets()`, all `const` pointers |

| Result type | Fields |
|-------------|--------|
| `FitResult` | `int iterations`, `double delta` |
| `DeltaFitResult` | `std::vector<std::pair<double, int64_t>> histogram`, `std::vector<double> multisets`, `int64_t num_subgraphs` |

CUDA: `#include "dress/cuda/dress.hpp"`; same class in `dress::cuda` namespace.

---

## Python

Package: `dress`

**Free functions** (C extension or pure-Python fallback):

| Function | Signature |
|----------|-----------|
| `fit` | `fit(n_vertices, sources, targets, weights=None, variant=UNDIRECTED, max_iterations=100, epsilon=1e-6, precompute_intercepts=False) → DRESSResult` |
| `delta_fit` | `delta_fit(n_vertices, sources, targets, weights=None, k=0, variant=UNDIRECTED, max_iterations=100, epsilon=1e-6, precompute=False, keep_multisets=False) → DeltaDRESSResult` |
| `nabla_fit` | `nabla_fit(n_vertices, sources, targets, weights=None, k=0, variant=UNDIRECTED, max_iterations=100, epsilon=1e-6, n_samples=0, seed=0, precompute=False, keep_multisets=False) → NablaDRESSResult` |

| Result class | Fields |
|--------------|--------|
| `DRESSResult` | `sources: list[int]`, `targets: list[int]`, `edge_dress: list[float]`, `edge_weight: list[float]`, `vertex_dress: list[float]`, `iterations: int`, `delta: float` |
| `DeltaDRESSResult` | `histogram: list[tuple[float, int]]`, `multisets: object \| None`, `num_subgraphs: int` |
| `NablaDRESSResult` | `histogram: list[tuple[float, int]]`, `multisets: object \| None`, `num_tuples: int` |
| `FitResult` | `iterations: int`, `delta: float` |

**`DRESS` class** (persistent graph):

| Method | Signature |
|--------|-----------|
| Constructor | `DRESS(n_vertices, sources, targets, weights=None, variant=UNDIRECTED, precompute_intercepts=False)` |
| `fit` | `fit(max_iterations=100, epsilon=1e-6) → FitResult` |
| `get` | `get(u, v, max_iterations=100, epsilon=1e-6, edge_weight=1.0) → float` |
| Properties | `n_vertices`, `n_edges`, `edge_dress(e)`, `edge_weight(e)`, `vertex_dress(u)` |

| Variant constants | `UNDIRECTED`, `DIRECTED`, `FORWARD`, `BACKWARD` |
|---|---|

CUDA: `from dress.cuda import fit, delta_fit, nabla_fit, DRESS`
MPI: `from dress.mpi import delta_fit, nabla_fit`
MPI+CUDA: `from dress.mpi.cuda import delta_fit, nabla_fit`

### NetworkX wrappers (Python)

| Backend | Import path | Exports |
|---------|-------------|---------|
| CPU | `dress.networkx` | `fit`, `delta_fit`, `nabla_fit`, `NxDRESS` |
| OpenMP | `dress.omp.networkx` | `fit`, `delta_fit`, `nabla_fit`, `NxDRESS` |
| CUDA | `dress.cuda.networkx` | `fit`, `delta_fit`, `nabla_fit`, `NxDRESS` |
| MPI | `dress.mpi.networkx` | `delta_fit`, `nabla_fit` |
| MPI+OpenMP | `dress.mpi.omp.networkx` | `delta_fit`, `nabla_fit` |
| MPI+CUDA | `dress.mpi.cuda.networkx` | `delta_fit`, `nabla_fit` |

| Function / Class | Signature |
|----------|-----------|
| `fit` | `fit(G, *, variant=UNDIRECTED, weight="weight", max_iterations=100, epsilon=1e-6, precompute_intercepts=False, set_attributes=False) → DRESSResult` |
| `delta_fit` | `delta_fit(G, *, k=0, variant=UNDIRECTED, weight="weight", max_iterations=100, epsilon=1e-6, precompute=False, keep_multisets=False) → DeltaDRESSResult` |
| `nabla_fit` | `nabla_fit(G, *, k=0, variant=UNDIRECTED, weight="weight", max_iterations=100, epsilon=1e-6, n_samples=0, seed=0, precompute=False, keep_multisets=False) → NablaDRESSResult` |
| `NxDRESS` | `NxDRESS(G, *, variant=UNDIRECTED, weight="weight", precompute_intercepts=False)` |
| `NxDRESS.fit` | `fit(max_iterations=100, epsilon=1e-6) → FitResult` |
| `NxDRESS.get` | `get(u, v, max_iterations=100, epsilon=1e-6, edge_weight=1.0) → float` |
| `NxDRESS.result` | `result() → DRESSResult` |
| `NxDRESS.close` | `close()` |

MPI wrappers accept an additional `comm=None` keyword (defaults to `MPI.COMM_WORLD`).

When `set_attributes=True`, `fit` writes `"dress"` edge attributes and `"vertex_dress"` node attributes back onto the NetworkX graph.

---

## Rust

Crate: `dress-graph`

**Builder API:**

| Method | Signature |
|--------|-----------|
| `DRESS::builder` | `DRESS::builder(n: i32, sources: Vec<i32>, targets: Vec<i32>) → DRESSBuilder` |
| `.weights()` | `.weights(w: Vec<f64>) → DRESSBuilder` |
| `.variant()` | `.variant(v: Variant) → DRESSBuilder` |
| `.max_iterations()` | `.max_iterations(n: i32) → DRESSBuilder` |
| `.epsilon()` | `.epsilon(e: f64) → DRESSBuilder` |
| `.precompute_intercepts()` | `.precompute_intercepts(b: bool) → DRESSBuilder` |
| `.build()` | `.build() → Result<DRESS, DressError>` |
| `.build_and_fit()` | `.build_and_fit() → Result<DressResult, DressError>` |

**`DRESS` persistent graph:**

| Method | Signature |
|--------|-----------|
| `fit` | `fn fit(&mut self, max_iterations: i32, epsilon: f64) → (i32, f64)` |
| `get` | `fn get(&self, u: i32, v: i32, max_iterations: i32, epsilon: f64, edge_weight: f64) → f64` |

**Free function:**

| Function | Signature |
|----------|-----------|
| `DRESS::delta_fit` | `fn delta_fit(n, sources, targets, weights: Option<Vec<f64>>, k, max_iterations, epsilon, variant: Variant, precompute: bool, keep_multisets: bool) → Result<DeltaDressResult, DressError>` |
| `DRESS::nabla_fit` | `fn nabla_fit(n, sources, targets, weights: Option<Vec<f64>>, k, max_iterations, epsilon, n_samples: i32, seed: u32, variant: Variant, precompute: bool, keep_multisets: bool) → Result<NablaDressResult, DressError>` |

| Result type | Fields |
|-------------|--------|
| `DressResult` | `sources: Vec<i32>`, `targets: Vec<i32>`, `edge_dress: Vec<f64>`, `edge_weight: Vec<f64>`, `vertex_dress: Vec<f64>`, `iterations: i32`, `delta: f64` |
| `HistogramEntry` | `value: f64`, `count: i64` |
| `DeltaDressResult` | `histogram: Vec<HistogramEntry>`, `multisets: Option<Vec<f64>>`, `num_subgraphs: i64` |
| `NablaDressResult` | `histogram: Vec<HistogramEntry>`, `multisets: Option<Vec<f64>>`, `num_tuples: i64` |
| `Variant` | `Undirected`, `Directed`, `Forward`, `Backward` |

CUDA: `use fit::cuda::DRESS` ; same API.

---

## Go

Package: `github.com/velicast/dress-graph/go`

| Function | Signature |
|----------|-----------|
| `Fit` | `func Fit(n int, sources, targets []int32, weights []float64, variant Variant, maxIterations int, epsilon float64, precomputeIntercepts bool) (*Result, error)` |
| `DeltaFit` | `func DeltaFit(n int, sources, targets []int32, weights []float64, k int, variant Variant, maxIterations int, epsilon float64, nSamples int, seed uint32, precompute bool, keepMultisets bool, computeHistogram bool) (*DeltaResult, error)` |
| `NablaFit` | `func NablaFit(n int, sources, targets []int32, weights []float64, k int, variant Variant, maxIterations int, epsilon float64, nSamples int, seed uint32, precompute bool, keepMultisets bool, computeHistogram bool) (*NablaResult, error)` |

**`DRESS` persistent graph:**

| Method | Signature |
|--------|-----------|
| `NewDRESS` | `func NewDRESS(n int, sources, targets []int32, weights []float64, variant Variant, precomputeIntercepts bool) (*DRESS, error)` |
| `Fit` | `func (dg *DRESS) Fit(maxIterations int, epsilon float64) (iterations int, delta float64, err error)` |
| `Get` | `func (dg *DRESS) Get(u, v int, maxIterations int, epsilon float64, edgeWeight float64) (float64, error)` |
| `Result` | `func (dg *DRESS) Result() (*Result, error)` |
| `Close` | `func (dg *DRESS) Close()` |

| Result type | Fields |
|-------------|--------|
| `Result` | `Sources []int32`, `Targets []int32`, `EdgeDress []float64`, `EdgeWeight []float64`, `VertexDress []float64`, `Iterations int`, `Delta float64` |
| `HistogramEntry` | `Value float64`, `Count int64` |
| `DeltaResult` | `Histogram []HistogramEntry`, `Multisets []float64` (nil when not requested), `NumSubgraphs int64` |
| `NablaResult` | `Histogram []HistogramEntry`, `Multisets []float64` (nil when not requested), `NumTuples int64` |
| `Variant` | `Undirected`, `Directed`, `Forward`, `Backward` |

CUDA: `import dress "github.com/velicast/dress-graph/go/cuda"` ; same API.

---

## JavaScript / WASM

Module: `dress.js`

| Function | Signature |
|----------|-----------|
| `fit` | `async fit(opts: DressOptions) → DressResult` |
| `deltaFit` | `async deltaFit(opts: DeltaDressOptions) → DeltaDressResult` |
| `nablaFit` | `async nablaFit(opts: NablaDressOptions) → NablaDressResult` |

| Options type | Fields |
|-------------|--------|
| `DressOptions` | `numVertices: number`, `sources: Int32Array \| number[]`, `targets: Int32Array \| number[]`, `weights?: Float64Array \| number[] \| null`, `variant?: number`, `maxIterations?: number`, `epsilon?: number`, `precomputeIntercepts?: boolean` |
| `DeltaDressOptions` | `numVertices`, `sources`, `targets`, `weights?`, `k?: number`, `variant?`, `maxIterations?`, `epsilon?`, `precompute?: boolean`, `keepMultisets?: boolean` |

| Result type | Fields |
|-------------|--------|
| `DressResult` | `sources: Int32Array`, `targets: Int32Array`, `edgeDress: Float64Array`, `edgeWeight: Float64Array`, `vertexDress: Float64Array`, `iterations: number`, `delta: number` |
| `HistogramEntry` | `{ value: number, count: number }` |
| `DeltaDressResult` | `histogram: HistogramEntry[]`, `multisets: Float64Array \| null`, `numSubgraphs: number` |
| `NablaDressResult` | `histogram: HistogramEntry[]`, `multisets: Float64Array \| null`, `numTuples: number` |

**`DRESS` class:**

| Method | Signature |
|--------|-----------|
| `create` | `static async create(opts: DRESSOptions) → DRESS` |
| `fit` | `fit(maxIterations?: number, epsilon?: number) → { iterations: number, delta: number }` |
| `get` | `get(u: number, v: number, maxIterations?: number, epsilon?: number, edgeWeight?: number) → number` |
| `result` | `result() → DressResult` |
| `free` | `free() → void` |

---

## Julia

Package: `DRESS`

| Function | Signature |
|----------|-----------|
| `fit` | `fit(N, sources, targets; weights=nothing, variant=UNDIRECTED, max_iterations=100, epsilon=1e-6, precompute_intercepts=false) → DRESSResult` |
| `delta_fit` | `delta_fit(N, sources, targets; weights=nothing, k=0, variant=UNDIRECTED, max_iterations=100, epsilon=1e-6, precompute=false, keep_multisets=false) → DeltaDRESSResult` |
| `nabla_fit` | `nabla_fit(N, sources, targets; weights=nothing, k=0, variant=UNDIRECTED, max_iterations=100, epsilon=1e-6, n_samples=0, seed=0, precompute=false, keep_multisets=false) → NablaDRESSResult` |

**`DressGraph` persistent object:**

| Function | Signature |
|----------|-----------|
| Constructor | `DressGraph(N, sources, targets; weights=nothing, variant=UNDIRECTED, precompute_intercepts=false)` |
| `fit!` | `fit!(g; max_iterations=100, epsilon=1e-6) → (iterations::Int, delta::Float64)` |
| `get` | `get(g, u, v; max_iterations=100, epsilon=1e-6, edge_weight=1.0) → Float64` |
| `close` | `close(g)` |

| Result type | Fields |
|-------------|--------|
| `DRESSResult` | `sources::Vector{Int32}`, `targets::Vector{Int32}`, `edge_weight::Vector{Float64}`, `edge_dress::Vector{Float64}`, `vertex_dress::Vector{Float64}`, `iterations::Int`, `delta::Float64` |
| `HistogramEntry` | `value::Float64`, `count::Int64` |
| `DeltaDRESSResult` | `histogram::Vector{HistogramEntry}`, `multisets::Union{Matrix{Float64}, Nothing}`, `num_subgraphs::Int` |
| `NablaDRESSResult` | `histogram::Vector{HistogramEntry}`, `multisets::Union{Matrix{Float64}, Nothing}`, `num_tuples::Int` |
| Variants | `UNDIRECTED`, `DIRECTED`, `FORWARD`, `BACKWARD` |

CUDA: `using DRESS.CUDA` ; same functions.

---

## R

Package: `dress.graph`

| Function | Signature |
|----------|-----------|
| `fit` | `fit(n_vertices, sources, targets, weights=NULL, variant=DRESS_UNDIRECTED, max_iterations=100L, epsilon=1e-6, precompute_intercepts=FALSE) → list` |
| `delta_fit` | `delta_fit(n_vertices, sources, targets, weights=NULL, k=0L, variant=DRESS_UNDIRECTED, max_iterations=100L, epsilon=1e-6, precompute=FALSE, keep_multisets=FALSE) → list` |
| `nabla_fit` | `nabla_fit(n_vertices, sources, targets, weights=NULL, k=0L, variant=DRESS_UNDIRECTED, max_iterations=100L, epsilon=1e-6, n_samples=0L, seed=0L, precompute=FALSE, keep_multisets=FALSE) → list` |

**`DRESS` persistent object:**

| Method | Signature |
|--------|-----------|
| Constructor | `DRESS(n_vertices, sources, targets, weights=NULL, variant=DRESS_UNDIRECTED, precompute_intercepts=FALSE)` |
| `$fit` | `$fit(max_iterations=100L, epsilon=1e-6) → list(iterations, delta)` |
| `$get` | `$get(u, v, max_iterations=100L, epsilon=1e-6, edge_weight=1.0) → numeric` |
| `$result` | `$result() → list` |
| `$close` | `$close()` |

`fit` returns a list with: `$sources`, `$targets`, `$edge_dress`, `$edge_weight`, `$vertex_dress`, `$iterations`, `$delta`.

`delta_fit` returns a list with: `$histogram` (data frame with `value` and `count` columns), `$multisets` (matrix, when `keep_multisets=TRUE`), `$num_subgraphs` (when `keep_multisets=TRUE`).

| Variant constants | `DRESS_UNDIRECTED`, `DRESS_DIRECTED`, `DRESS_FORWARD`, `DRESS_BACKWARD` |
|---|---|

CUDA: `cuda$fit(...)`, `cuda$delta_fit(...)`, `cuda$nabla_fit(...)`.
MPI: `mpi$delta_fit(...)`, `mpi$nabla_fit(...)`.
MPI+CUDA: `mpi$cuda$delta_fit(...)`, `mpi$cuda$nabla_fit(...)`.

---

## MATLAB / Octave

CPU and CUDA expose both functional and OO APIs. MPI and MPI+CUDA are available through persistent objects in the `+mpi/` and `+mpi/+cuda/` namespaces.

| Function | Signature |
|----------|-----------|
| `fit` | `fit(n_vertices, sources, targets, 'Weights', [], 'Variant', 0, 'MaxIterations', 100, 'Epsilon', 1e-6, 'PrecomputeIntercepts', false) → struct` |
| `delta_fit` | `delta_fit(n_vertices, sources, targets, 'K', 0, 'Variant', 0, 'MaxIterations', 100, 'Epsilon', 1e-6, 'Precompute', false, 'KeepMultisets', false) → struct` |
| `nabla_fit` | `nabla_fit(n_vertices, sources, targets, 'K', 0, 'Variant', 0, 'MaxIterations', 100, 'Epsilon', 1e-6, 'NSamples', 0, 'Seed', 0, 'Precompute', false, 'KeepMultisets', false) → struct` |

**Persistent graph classes (handle):**

| Class | Backend | Delta method |
|-------|---------|--------------|
| `DRESS` | CPU | none |
| `omp.DRESS` | OpenMP | none |
| `cuda.DRESS` | CUDA | none |
| `mpi.DRESS` | MPI | `delta_fit(...)` |
| `mpi.omp.DRESS` | MPI+OpenMP | `delta_fit(...)` |
| `mpi.cuda.DRESS` | MPI+CUDA | `delta_fit(...)` |

**Common object methods:**

| Method | Signature |
|--------|-----------|
| Constructor | `DRESS(n_vertices, sources, targets, 'Weights', [], ...)` |
| `fit` | `g.fit('MaxIterations', 100, 'Epsilon', 1e-6) → struct(iterations, delta)` |
| `get` | `g.get(u, v, 'MaxIterations', 100, 'Epsilon', 1e-6, 'EdgeWeight', 1.0) → double` |
| `result` | `g.result() → struct` |
| `close` | `g.close()` |

`fit` returns a struct with: `.sources`, `.targets`, `.edge_dress`, `.edge_weight`, `.vertex_dress`, `.iterations`, `.delta`.

`delta_fit` returns a struct with: `.histogram.value`, `.histogram.count`, `.multisets` (when `KeepMultisets=true`), `.num_subgraphs` (when `KeepMultisets=true`).

`mpi.DRESS(...).delta_fit(...)` and `mpi.cuda.DRESS(...).delta_fit(...)` return the same sparse histogram struct shape as `delta_fit(...)`.

Variant values: `0` = undirected, `1` = directed, `2` = forward, `3` = backward.

CPU: `fit(...)`, `delta_fit(...)`, `nabla_fit(...)`, `DRESS(...)`.

CUDA: `cuda.fit(...)`, `cuda.delta_fit(...)`, `cuda.nabla_fit(...)`, `cuda.DRESS(...)` (MATLAB `+cuda/` namespace; Octave `+cuda/` namespace; requires `libdress_cuda.so`).

MPI: `mpi.DRESS(...).delta_fit(...)`, `mpi.DRESS(...).nabla_fit(...)` (MATLAB `+mpi/` namespace; Octave `+mpi/` namespace; requires MPI MEX support in the local build/package).

MPI+CUDA: `mpi.cuda.DRESS(...).delta_fit(...)` (MATLAB `+mpi/+cuda/` namespace; Octave `+mpi/+cuda/` namespace; requires MPI+CUDA MEX support in the local build/package).