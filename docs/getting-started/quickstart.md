# Quick Start

## Python

```python
from dress import dress_fit

result = dress_fit(
    n_vertices=4,
    sources=[0, 1, 2, 0],
    targets=[1, 2, 3, 3],
)

print(f"Iterations: {result.iterations}")
print(f"Edge DRESS values: {result.edge_dress}")
print(f"Node DRESS values: {result.node_dress}")
```

## Rust

```rust
use dress_graph::{DRESS, Variant};

let result = DRESS::builder(4, vec![0, 1, 2, 0], vec![1, 2, 3, 3])
    .variant(Variant::Undirected)
    .max_iterations(100)
    .epsilon(1e-6)
    .build_and_fit()
    .unwrap();

println!("iterations: {}", result.iterations);
for (i, d) in result.edge_dress.iter().enumerate() {
    println!("  edge {}: dress = {:.6}", i, d);
}
```

## C

```c
#include <stdio.h>
#include <stdlib.h>
#include "dress/dress.h"

int main(void) {
    int N = 4, E = 4;

    /* Allocate with malloc. init_dress_graph takes ownership. */
    int    *U = malloc(E * sizeof *U);
    int    *V = malloc(E * sizeof *V);
    U[0] = 0; V[0] = 1;
    U[1] = 1; V[1] = 2;
    U[2] = 2; V[2] = 3;
    U[3] = 0; V[3] = 3;

    p_dress_graph_t g = init_dress_graph(N, E, U, V, NULL,
                                         DRESS_VARIANT_UNDIRECTED, 1);

    int iterations;
    double delta;
    fit(g, 100, 1e-6, &iterations, &delta);

    printf("iterations: %d, delta: %e\n", iterations, delta);
    for (int e = 0; e < g->E; e++)
        printf("  edge (%d,%d): dress = %.6f\n",
               g->U[e], g->V[e], g->edge_dress[e]);

    free_dress_graph(g);
    return 0;
}
```

## C++

```cpp
#include <cstdio>
#include "dress/dress.hpp"

int main() {
    DRESS g(4,
            {0, 1, 2, 0},   // sources
            {1, 2, 3, 3},   // targets
            DRESS_VARIANT_UNDIRECTED,
            true);           // precompute intercepts

    auto [iters, delta] = g.fit(100, 1e-6);

    printf("iterations: %d, delta: %e\n", iters, delta);
    for (int e = 0; e < g.numEdges(); e++)
        printf("  edge (%d,%d): dress = %.6f\n",
               g.edgeSource(e), g.edgeTarget(e), g.edgeDress(e));
}
```

## Go

```go
import "github.com/velicast/dress-graph/go"

result, err := dress.Fit(4,
    []int32{0, 1, 2, 0},
    []int32{1, 2, 3, 3},
    nil, // no weights
    dress.Undirected, 100, 1e-6, true)

fmt.Printf("iterations: %d\n", result.Iterations)
```

## JavaScript (WASM)

```javascript
import { dressFit, Variant } from './dress.js';

const result = await dressFit({
    numVertices: 4,
    sources: [0, 1, 2, 0],
    targets: [1, 2, 3, 3],
    variant: Variant.UNDIRECTED,
});

console.log(`iterations: ${result.iterations}`);
console.log('edge dress:', result.edgeDress);
```

## R

```r
library(dress.graph)

res <- dress_fit(
  n_vertices = 4L,
  sources    = c(0L, 1L, 2L, 0L),
  targets    = c(1L, 2L, 3L, 3L)
)

cat("iterations:", res$iterations, "\n")
cat("edge dress:", res$edge_dress, "\n")
cat("node dress:", res$node_dress, "\n")
```

## Julia

```julia
using DRESS

result = dress_fit(4, Int32[0, 1, 2, 0], Int32[1, 2, 3, 3];
                   variant=UNDIRECTED)

println("iterations: ", result.iterations)
println("edge dress: ", result.edge_dress)
```

## MATLAB / Octave

```matlab
result = dress_fit(4, int32([0 1 2 0]), int32([1 2 3 3]));

fprintf('iterations: %d\n', result.iterations);
disp(result.edge_dress);
```
