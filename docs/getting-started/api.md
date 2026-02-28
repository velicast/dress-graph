# API Reference

## Core concepts

All bindings expose the same underlying algorithm through three main function
calls: **`dress_fit()`** and **`delta_dress_fit()`**.

**`dress_fit()`** — Pass in the graph (vertices, edges, optional
weights), and get back a result struct containing every output array.

```
result = dress_fit(n_vertices, sources, targets, [weights], [variant], ...)
```

The result always contains the same fields across every language:
edge dress values, edge weights, node dress norms, iteration count, and
final convergence delta.

**`delta_dress_fit()`** — Compute the \(\Delta^k\)-DRESS histogram.
Enumerates all \(\binom{N}{k}\) vertex-deletion subsets, runs DRESS on
each subgraph, and accumulates edge values into a histogram.

```
result = delta_dress_fit(n_vertices, sources, targets, [k], [epsilon], ...)
```

`delta_dress_fit` returns a histogram of integer counts and its size.
Each bin \(i\) counts edge values in \([i \cdot \varepsilon,\; (i+1) \cdot \varepsilon)\),
with the top bin (index \(\lfloor 2/\varepsilon \rfloor\)) holding
exact value 2.0.  The number of bins is \(\lfloor 2/\varepsilon \rfloor + 1\).

## Variants

| Constant | Value | Neighbourhood \(N[u]\) | Combined weight \(\bar{w}(u,v)\) |
|----------|-------|------------------------|----------------------------------|
| `UNDIRECTED` | 0 | \(\{u\} \cup\) all neighbors (ignoring direction) | \(2\,w(u,v)\) |
| `DIRECTED` | 1 | \(\{u\} \cup\) all neighbors (in + out) | \(w(u,v) + w(v,u)\) |
| `FORWARD` | 2 | \(\{u\} \cup\) out-neighbors | \(w(u,v)\) |
| `BACKWARD` | 3 | \(\{u\} \cup\) in-neighbors | \(w(v,u)\) |

## Result fields

### `dress_fit` result

| Field | Type | Description |
|-------|------|-------------|
| `sources` | int array [E] | Edge source vertices (0-based) |
| `targets` | int array [E] | Edge target vertices (0-based) |
| `edge_dress` | float array [E] | DRESS similarity per edge |
| `edge_weight` | float array [E] | Variant-specific edge weight |
| `node_dress` | float array [N] | Per-node aggregated DRESS norm |
| `iterations` | int | Number of iterations performed |
| `delta` | float | Final maximum per-edge change |

### `delta_dress_fit` result

| Field | Type | Description |
|-------|------|-------------|
| `histogram` | int64 array [hist_size] | Bin counts of converged edge values |
| `hist_size` | int | Number of bins: \(\lfloor 2/\varepsilon \rfloor + 1\) |

## Language-specific APIs

### C

```c
#include "dress/dress.h"

/* Variant enum */
typedef enum {
    DRESS_VARIANT_UNDIRECTED = 0,
    DRESS_VARIANT_DIRECTED   = 1,
    DRESS_VARIANT_FORWARD    = 2,
    DRESS_VARIANT_BACKWARD   = 3
} dress_variant_t;

/* Construct a dress graph from an edge list.
   Takes ownership of U, V, W (freed by free_dress_graph).
   W may be NULL for unweighted graphs. */
p_dress_graph_t init_dress_graph(int N, int E,
                                 int *U, int *V, double *W,
                                 dress_variant_t variant,
                                 int precompute_intercepts);

/* Run iterative fitting. */
void fit(p_dress_graph_t g, int max_iterations, double epsilon,
         int *iterations, double *delta);

/* Free all memory. */
void free_dress_graph(p_dress_graph_t g);

/* Result fields on the struct: */
g->edge_dress[e]   /* per-edge DRESS value */
g->edge_weight[e]  /* per-edge weight */
g->node_dress[u]   /* per-node norm */
g->U[e], g->V[e]   /* edge endpoints */
g->N, g->E         /* vertex / edge count */
```

#### Δ^k-DRESS (C)

```c
#include "dress/delta_dress.h"

/* Compute the Δ^k-DRESS histogram.
   Returns a malloc'd int64_t array; caller must free(). */
int64_t *delta_fit(p_dress_graph_t g,
                   int k,              /* deletion depth (0 = original) */
                   int iterations,     /* max DRESS iterations per subgraph */
                   double epsilon,     /* convergence tol / bin width */
                   int *hist_size,     /* [out] number of bins */
                   int keep_multisets,     /* if non-zero, allocate multisets */
                   double **multisets,     /* [out] internally-allocated C(N,k)*E, or NULL */
                   int64_t *num_subgraphs);/* [out] C(N,k), or NULL */
```

### C (igraph wrapper)

```c
#include "dress_igraph.h"

/* Result structure (pointers are zero-copy views into internal data) */
typedef struct {
    int     N;           /* number of vertices                       */
    int     E;           /* number of edges                          */
    const int    *src;   /* [E] edge source endpoints                */
    const int    *dst;   /* [E] edge target endpoints                */
    const double *dress; /* [E] DRESS similarity per edge            */
    const double *weight;/* [E] variant edge weight                  */
    const double *node_dress; /* [N] per-node DRESS norm             */
    int     iterations;  /* iterations performed                     */
    double  delta;       /* final max-delta at convergence           */
} dress_igraph_result_t;

/* Compute DRESS on an igraph graph.
   weight_attr: edge attribute name (e.g. "weight"), or NULL.
   Returns 0 on success. */
int dress_igraph_compute(const igraph_t *graph,
                         const char *weight_attr,
                         dress_variant_t variant,
                         int max_iters, double epsilon,
                         int precompute,
                         dress_igraph_result_t *result);

/* Free internal memory (struct itself is not freed). */
void dress_igraph_free(dress_igraph_result_t *result);

/* Copy per-edge DRESS values into an igraph_vector_t. */
int dress_igraph_to_vector(const dress_igraph_result_t *result,
                           igraph_vector_t *out);
```

Build with:

```bash
gcc -O2 my_app.c dress_igraph.c dress.c \
    $(pkg-config --cflags --libs igraph) -lm -fopenmp
```

Or via CMake:

```bash
cmake -B build -DDRESS_BUILD_IGRAPH=ON
cmake --build build
```

### C++

```cpp
#include "dress/dress.hpp"

/* RAII wrapper. Copies vectors into malloc'd buffers automatically. */

/* Unweighted */
DRESS g(N, sources_vec, targets_vec,
        DRESS_VARIANT_UNDIRECTED, precompute_intercepts);

/* Weighted */
DRESS g(N, sources_vec, targets_vec, weights_vec,
        DRESS_VARIANT_UNDIRECTED, precompute_intercepts);

/* Fit and read results */
auto [iterations, delta] = g.fit(max_iterations, epsilon);

g.numVertices()        // int
g.numEdges()           // int
g.edgeDress(e)         // double, DRESS value for edge e
g.edgeWeight(e)        // double
g.nodeDress(u)         // double
g.edgeSource(e)        // int
g.edgeTarget(e)        // int

/* Bulk zero-copy access */
g.edgeDressValues()    // const double*
g.edgeWeights()        // const double*
g.nodeDressValues()    // const double*
g.edgeSources()        // const int*
g.edgeTargets()        // const int*
```

#### Δ^k-DRESS (C++)

```cpp
// Method on the DRESS class
DRESS::DeltaFitResult deltaFit(int k, int maxIterations,
                                double epsilon,
                                bool keepMultisets = false);

// Result struct
struct DeltaFitResult {
    std::vector<int64_t> histogram;   // bin counts
    int                  hist_size;   // floor(2/epsilon) + 1
    std::vector<double>  multisets;   // C(N,k)*E row-major, NaN = removed
    int64_t              num_subgraphs; // C(N,k)
};
```

### Python

```python
from dress import dress_fit, UNDIRECTED, DIRECTED, FORWARD, BACKWARD

# Unweighted
result = dress_fit(n_vertices, sources, targets)

# Weighted
result = dress_fit(n_vertices, sources, targets,
                   weights=[1.0, 2.0, 3.0])

# With options
result = dress_fit(n_vertices, sources, targets,
                   variant=DIRECTED,
                   max_iterations=200,
                   epsilon=1e-12)

result.sources      # list[int]
result.targets      # list[int]
result.edge_dress   # list[float]
result.edge_weight  # list[float]
result.node_dress   # list[float]
result.iterations   # int
result.delta        # float
```

The `dress_fit()` function works identically whether the C extension
or the pure-Python backend is active.  It returns a single `DRESSResult`
with all arrays and metadata.

#### Δ^k-DRESS (Python)

```python
from dress import delta_dress_fit

result = delta_dress_fit(
    n_vertices, sources, targets,
    k=1,                    # deletion depth (default 0)
    variant=UNDIRECTED,     # graph variant
    max_iterations=100,     # per-subgraph iterations
    epsilon=1e-3,           # convergence tol / bin width
    precompute=False,       # precompute intercepts
)

result.histogram    # list[int], length = hist_size
result.hist_size    # int, floor(2/epsilon) + 1
```

The same function is available in pure Python via `from dress.core import delta_dress_fit`.

For advanced use (e.g. re-fitting with different parameters), the
low-level `DRESS` class is also available:

```python
from dress import DRESS, UNDIRECTED

g = DRESS(n_vertices, sources, targets, variant=UNDIRECTED)
fit_info = g.fit(max_iterations=100, epsilon=1e-6)
fit_info.iterations   # int
fit_info.delta        # float

# Read values from graph object
g.edge_dress(e)       # float, DRESS value for edge e
g.edge_weight(e)      # float
g.node_dress(u)       # float
g.n_vertices          # int
g.n_edges             # int
```

### Python (pure, no C dependencies)

```python
from dress.core import dress_fit, UNDIRECTED

result = dress_fit(n_vertices, sources, targets,
                   variant=UNDIRECTED)

result.sources      # list[int]
result.targets      # list[int]
result.edge_dress   # list[float]
result.edge_weight  # list[float]
result.node_dress   # list[float]
result.iterations   # int
result.delta        # float
```

The pure-Python backend is used automatically when the C extension is not
available (`import dress` falls back to `dress.core`).

#### NetworkX integration

```python
import networkx as nx
from dress.networkx import dress_graph

G = nx.karate_club_graph()
result = dress_graph(G, variant=UNDIRECTED,
                     max_iterations=100, epsilon=1e-6)

# Or write attributes back onto the graph:
result = dress_graph(G, set_attributes=True)
G.edges[0, 1]["dress"]      # per-edge similarity
G.nodes[0]["dress_norm"]     # per-node norm
```

### Rust

```rust
use dress_graph::{DRESS, Variant, DressResult, DressError};

let result: Result<DressResult, DressError> =
    DRESS::builder(n, sources, targets)
        .weights(weights)                   // optional
        .variant(Variant::Undirected)       // default
        .max_iterations(100)                // default
        .epsilon(1e-6)                      // default
        .precompute_intercepts(true)        // default
        .build_and_fit();

let r = result.unwrap();
r.sources       // Vec<i32>
r.targets       // Vec<i32>
r.edge_dress    // Vec<f64>
r.edge_weight   // Vec<f64>
r.node_dress    // Vec<f64>
r.iterations    // i32
r.delta         // f64
```

#### Δ^k-DRESS (Rust)

```rust
use dress_graph::{DRESS, Variant, DeltaDressResult, DressError};

let result: Result<DeltaDressResult, DressError> =
    DRESS::delta_fit(
        n, sources, targets,
        k,                          // deletion depth
        max_iterations,             // per-subgraph
        epsilon,                    // convergence tol / bin width
        Variant::Undirected,
        precompute,                 // bool
    );

let r = result.unwrap();
r.histogram     // Vec<i64>
r.hist_size     // i32
```

### Go

```go
import dress "github.com/velicast/dress-graph/go"

result, err := dress.Fit(
    n,                      // int, number of vertices
    sources,                // []int32
    targets,                // []int32
    weights,                // []float64 or nil
    dress.Undirected,       // Variant
    100,                    // maxIterations
    1e-6,                   // epsilon
    true,                   // precomputeIntercepts
)

result.Sources     // []int32
result.Targets     // []int32
result.EdgeDress   // []float64
result.EdgeWeight  // []float64
result.NodeDress   // []float64
result.Iterations  // int
result.Delta       // float64
```

#### Δ^k-DRESS (Go)

```go
result, err := dress.DeltaFit(
    n,                      // int, number of vertices
    sources,                // []int32
    targets,                // []int32
    k,                      // int, deletion depth
    dress.Undirected,       // Variant
    100,                    // maxIterations
    1e-3,                   // epsilon (bin width)
    false,                  // precompute
)

result.Histogram   // []int64
result.HistSize    // int
```

### JavaScript (WASM)

```javascript
import { dressFit, Variant } from './dress.js';

const result = await dressFit({
    numVertices: n,
    sources,                              // Int32Array or number[]
    targets,                              // Int32Array or number[]
    weights,                              // Float64Array, number[], or null
    variant: Variant.UNDIRECTED,          // default
    maxIterations: 100,                   // default
    epsilon: 1e-6,                        // default
    precomputeIntercepts: true,           // default
});

result.sources     // Int32Array
result.targets     // Int32Array
result.edgeDress   // Float64Array
result.edgeWeight  // Float64Array
result.nodeDress   // Float64Array
result.iterations  // number
result.delta       // number
```

#### Δ^k-DRESS (WASM)

```javascript
import { deltaDressFit } from './dress.js';

const result = await deltaDressFit({
    numVertices: n,
    sources,                              // Int32Array or number[]
    targets,                              // Int32Array or number[]
    k: 1,                                 // deletion depth (default 0)
    variant: Variant.UNDIRECTED,          // default
    maxIterations: 100,                   // default
    epsilon: 1e-3,                        // bin width
    precompute: false,                    // default
});

result.histogram   // Float64Array (int64 counts cast to double)
result.histSize    // number
```

### Julia

```julia
using DRESS

result = dress_fit(N, sources, targets;
                   weights = nothing,           # optional Vector{Float64}
                   variant = UNDIRECTED,         # UNDIRECTED/DIRECTED/FORWARD/BACKWARD
                   max_iterations = 100,
                   epsilon = 1e-6,
                   precompute_intercepts = true)

result.sources      # Vector{Int32}
result.targets      # Vector{Int32}
result.edge_dress   # Vector{Float64}
result.edge_weight  # Vector{Float64}
result.node_dress   # Vector{Float64}
result.iterations   # Int
result.delta        # Float64
```

#### Δ^k-DRESS (Julia)

```julia
result = delta_dress_fit(N, sources, targets;
                         k = 1,                     # deletion depth
                         variant = UNDIRECTED,
                         max_iterations = 100,
                         epsilon = 1e-3,
                         precompute = false)

result.histogram    # Vector{Int64}
result.hist_size    # Int
```

### R

```r
library(dress.graph)

result <- dress_fit(
  n_vertices,
  sources,                    # integer vector (0-based)
  targets,                    # integer vector (0-based)
  weights = NULL,             # optional numeric vector
  variant = DRESS_UNDIRECTED, # DRESS_UNDIRECTED / DRESS_DIRECTED /
                              # DRESS_FORWARD / DRESS_BACKWARD
  max_iterations = 100L,
  epsilon = 1e-6,
  precompute_intercepts = FALSE
)

result$sources      # integer [E]
result$targets      # integer [E]
result$edge_dress   # numeric [E]
result$edge_weight  # numeric [E]
result$node_dress   # numeric [N]
result$iterations   # integer
result$delta        # numeric
```

#### Δ^k-DRESS (R)

```r
result <- delta_dress_fit(
  n_vertices,
  sources,
  targets,
  k              = 1L,              # deletion depth
  variant        = DRESS_UNDIRECTED,
  max_iterations = 100L,
  epsilon        = 1e-3,            # bin width
  precompute     = FALSE
)

result$histogram    # numeric [hist_size]
result$hist_size    # integer
```

### MATLAB / Octave

```matlab
result = dress_fit(n_vertices, sources, targets, ...
    'Weights',              [],    ...   % double [E x 1] or []
    'Variant',              0,     ...   % 0=UNDIRECTED, 1=DIRECTED,
                                         % 2=FORWARD, 3=BACKWARD
    'MaxIterations',        100,   ...
    'Epsilon',              1e-6,  ...
    'PrecomputeIntercepts', false);

result.sources      % int32 [E x 1]
result.targets      % int32 [E x 1]
result.edge_dress   % double [E x 1]
result.edge_weight  % double [E x 1]
result.node_dress   % double [N x 1]
result.iterations   % int32
result.delta        % double
```

#### Δ^k-DRESS (MATLAB / Octave)

```matlab
result = delta_dress_fit(n_vertices, sources, targets, ...
    'K',             1,      ...   % deletion depth (default 0)
    'Variant',       0,      ...   % 0=UNDIRECTED
    'MaxIterations', 100,    ...
    'Epsilon',       1e-3,   ...   % bin width
    'Precompute',    false);

result.histogram    % double [hist_size x 1]
result.hist_size    % int32
```
