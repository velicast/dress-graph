# Quick Start

> **Note on Wrappers:** Please report any bugs you find while using the language wrappers. I am moving quickly and relying on AI to speed up the development of these wrappers, but I am actively and carefully maintaining the core C backend.

## Python

```python
from dress import fit

result = fit(
    n_vertices=4,
    sources=[0, 1, 2, 0],
    targets=[1, 2, 3, 3],
)

print(f"Iterations: {result.iterations}")
print(f"Edge DRESS values: {result.edge_dress}")
print(f"Node DRESS values: {result.node_dress}")
```

### Δ^k-DRESS (Python)

```python
from dress import delta_fit

result = delta_fit(
    n_vertices=4,
    sources=[0, 1, 2, 0],
    targets=[1, 2, 3, 3],
    k=1,            # remove 1 vertex at a time
    epsilon=1e-6,
)

print(result.histogram)  # [(value, count), ...]
print(f"Total edge-values: {sum(count for _, count in result.histogram)}")
print(f"Subgraphs: {result.num_subgraphs}")
```

### NetworkX (Python)

```python
import networkx as nx
from dress.networkx import fit, delta_fit

G = nx.karate_club_graph()

# Δ⁰: DRESS on the full graph
result = fit(G, set_attributes=True)
print(G.edges[0, 1]["dress"])     # per-edge similarity
print(G.nodes[0]["dress_norm"])   # per-node norm

# Δ¹: histogram fingerprint
delta = delta_fit(G, k=1)
print(delta.histogram)  # [(value, count), ...]
```

GPU and MPI variants, same API, different import:

```python
from dress.cuda.networkx import fit              # GPU
from dress.mpi.networkx import delta_fit          # MPI CPU
from dress.mpi.cuda.networkx import delta_fit     # MPI+CUDA
```

## Rust

```rust
use dress_graph::{DRESS, Variant};

let mut g = DRESS::new(4, vec![0, 1, 2, 0], vec![1, 2, 3, 3],
                       None, None, Variant::Undirected, false).unwrap();
let (iters, delta) = g.fit(100, 1e-6);
let r = g.result();

println!("iterations: {}", iters);
for (i, d) in r.edge_dress.iter().enumerate() {
    println!("  edge {}: dress = {:.6}", i, d);
}
```

### Δ^k-DRESS (Rust)

```rust
use dress_graph::{fit, delta_fit, Variant};

let r = delta_fit(
    4,
    vec![0, 1, 2, 0],
    vec![1, 2, 3, 3],
    None,     // edge weights
    None,     // node weights
    Variant::Undirected,
    false,    // precompute
    1,        // k = 1
    100,      // max iterations
    1e-6,
    0, 0,     // n_samples, seed
    false,    // keep multisets
    true,     // compute histogram
).unwrap();

println!("histogram entries: {:?}", r.histogram);
let total: i64 = r.histogram.iter().map(|entry| entry.count).sum();
println!("total edge-values: {}", total);
```

## C

Install via Homebrew, vcpkg, or build from source:

```bash
# Homebrew
brew tap velicast/dress-graph && brew install dress-graph

# vcpkg (overlay port)
vcpkg install dress-graph --overlay-ports=/path/to/dress-graph/vcpkg

# From source
mkdir build && cd build && cmake .. && make
```

```c
#include <stdio.h>
#include <stdlib.h>
#include "dress/dress.h"

int main(void) {
    int N = 4, E = 4;

    /* Allocate with malloc. dress_init_graph takes ownership. */
    int    *U = malloc(E * sizeof *U);
    int    *V = malloc(E * sizeof *V);
    U[0] = 0; V[0] = 1;
    U[1] = 1; V[1] = 2;
    U[2] = 2; V[2] = 3;
    U[3] = 0; V[3] = 3;

    p_dress_graph_t g = dress_init_graph(N, E, U, V, NULL, NULL,
                                         DRESS_VARIANT_UNDIRECTED, 1);

    int iterations;
    double delta;
    dress_fit(g, 100, 1e-6, &iterations, &delta);

    printf("iterations: %d, delta: %e\n", iterations, delta);
    for (int e = 0; e < g->E; e++)
        printf("  edge (%d,%d): dress = %.6f\n",
               g->U[e], g->V[e], g->edge_dress[e]);

    dress_free_graph(g);
    return 0;
}
```

### Δ^k-DRESS (C)

```c
#include "dress/dress.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int N = 4, E = 4;
    int *U = malloc(E * sizeof *U);
    int *V = malloc(E * sizeof *V);
    U[0]=0; V[0]=1; U[1]=1; V[1]=2;
    U[2]=2; V[2]=3; U[3]=0; V[3]=3;

    p_dress_graph_t g = dress_init_graph(N, E, U, V, NULL, NULL,
                                         DRESS_VARIANT_UNDIRECTED, 0);

    int hist_size;
    int64_t num_subgraphs;
    dress_hist_pair_t *hist = dress_delta_fit(g, 1, 100, 1e-6,
                                              0, 0,  // n_samples, seed
                                              &hist_size, 0, NULL, &num_subgraphs);

    int64_t total = 0;
    for (int i = 0; i < hist_size; i++) total += hist[i].count;
    printf("entries: %d, total: %ld\n", hist_size, total);

    free(hist);
    dress_free_graph(g);
    return 0;
}
```

## C++

```cpp
#include <cstdio>
#include "dress/dress.hpp"
using namespace dress;

int main() {
    DRESS g(4,
            {0, 1, 2, 0},   // sources
            {1, 2, 3, 3});  // targets

    auto [iters, delta] = g.fit(100, 1e-6);

    printf("iterations: %d, delta: %e\n", iters, delta);
    for (int e = 0; e < g.numEdges(); e++)
        printf("  edge (%d,%d): dress = %.6f\n",
               g.edgeSource(e), g.edgeTarget(e), g.edgeDress(e));
}
```

### Δ^k-DRESS (C++)

```cpp
#include <cstdio>
#include "dress/dress.hpp"
using namespace dress;

int main() {
    DRESS g(4, {0,1,2,0}, {1,2,3,3});
    auto r = g.deltaFit(1, 100, 1e-6);

    int64_t total = 0;
    for (const auto& [val, cnt] : r.histogram) total += cnt;
    printf("entries: %zu, total: %ld\n", r.histogram.size(), total);
}
```

## Go

```go
import "github.com/velicast/dress-graph/go"

result, err := dress.Fit(4,
    []int32{0, 1, 2, 0},
    []int32{1, 2, 3, 3},
    nil, // no weights
    nil, // no node weights
    dress.Undirected, 100, 1e-6, true)

fmt.Printf("iterations: %d\n", result.Iterations)
```

### Δ^k-DRESS (Go)

```go
r, err := dress.DeltaFit(4,
    []int32{0, 1, 2, 0},
    []int32{1, 2, 3, 3},
    nil,                 // edge weights
    nil,                 // node weights
    1,                   // k
    dress.Undirected,
    100, 1e-6,
    0, 0,                // n_samples, seed
    false, false, true)  // precompute, keepMultisets, computeHistogram

total := int64(0)
for _, entry := range r.Histogram { total += entry.Count }
fmt.Printf("entries: %d, total: %d\n", len(r.Histogram), total)
```

## JavaScript (WASM)

```javascript
import { fit, Variant } from './dress.js';

const result = await fit({
    numVertices: 4,
    sources: [0, 1, 2, 0],
    targets: [1, 2, 3, 3],
    variant: Variant.UNDIRECTED,
});

console.log(`iterations: ${result.iterations}`);
console.log('edge dress:', result.edgeDress);
```

### Δ^k-DRESS (WASM)

```javascript
import { deltaFit } from './dress.js';

const r = await deltaFit({
    numVertices: 4,
    sources: [0, 1, 2, 0],
    targets: [1, 2, 3, 3],
    k: 1,
    epsilon: 1e-6,
});

let total = 0;
for (const entry of r.histogram) total += entry.count;
console.log(r.histogram);
console.log(`total: ${total}`);
```

## R

```r
library(dress.graph)

res <- fit(
  n_vertices = 4L,
  sources    = c(0L, 1L, 2L, 0L),
  targets    = c(1L, 2L, 3L, 3L)
)

cat("iterations:", res$iterations, "\n")
cat("edge dress:", res$edge_dress, "\n")
cat("node dress:", res$node_dress, "\n")
```

### Δ^k-DRESS (R)

```r
r <- delta_fit(4L, c(0L,1L,2L,0L), c(1L,2L,3L,3L),
                     k = 1L, epsilon = 1e-6)
print(r$histogram)
cat("total:", sum(r$histogram$count), "\n")
```

## Julia

```julia
using DRESS

result = fit(4, Int32[0, 1, 2, 0], Int32[1, 2, 3, 3];
                   variant=UNDIRECTED)

println("iterations: ", result.iterations)
println("edge dress: ", result.edge_dress)
```

### Δ^k-DRESS (Julia)

```julia
r = delta_fit(4, Int32[0,1,2,0], Int32[1,2,3,3];
                    k=1, epsilon=1e-6)
println("histogram: ", r.histogram)
println("total: ", sum(entry.count for entry in r.histogram))
```

## MATLAB / Octave

```matlab
result = fit(4, int32([0 1 2 0]), int32([1 2 3 3]));

fprintf('iterations: %d\n', result.iterations);
disp(result.edge_dress);
```

### Δ^k-DRESS (MATLAB / Octave)

```matlab
r = delta_fit(4, int32([0 1 2 0]), int32([1 2 3 3]), ...
                    'K', 1, 'Epsilon', 1e-6);
disp(r.histogram.value);
disp(r.histogram.count);
```

OO and backend-switched variants use the same namespace pattern in both MATLAB and Octave:

```matlab
g  = DRESS(4, int32([0 1 2 0]), int32([1 2 3 3]));          % CPU
gc = cuda.DRESS(4, int32([0 1 2 0]), int32([1 2 3 3]));     % CUDA
gm = mpi.DRESS(4, int32([0 1 2 0]), int32([1 2 3 3]));      % MPI
gg = mpi.cuda.DRESS(4, int32([0 1 2 0]), int32([1 2 3 3])); % MPI+CUDA

r = gm.delta_fit('K', 1, 'KeepMultisets', true);
disp(r.histogram.value);
disp(r.histogram.count);
```
