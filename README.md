# dress-graph

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

PyPi [![PyPI Downloads](https://static.pepy.tech/personalized-badge/dress-graph?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/dress-graph) Crates [![crates.io downloads](https://img.shields.io/crates/d/dress-graph)](https://crates.io/crates/dress-graph) NPM [![npm downloads](https://img.shields.io/npm/dt/dress-graph)](https://www.npmjs.com/package/dress-graph) CRAN [![CRAN downloads](https://cranlogs.r-pkg.org/badges/grand-total/dress.graph)](https://cran.r-project.org/package=dress.graph)

### A Continuous Framework for Structural Graph Refinement

DRESS is a deterministic, parameter-free framework for continuous structural graph refinement. It iterates a nonlinear dynamical system on real-valued edge similarities and produces a graph fingerprint as a sorted edge-value vector once the iteration reaches a prescribed stopping criterion. The resulting fingerprint is self-contained, isomorphism-invariant by construction, reproducible across vertex labelings under the reference implementation, numerically robust in practice, and efficient to compute with straightforward parallelization and distribution.

```bash
pip install dress-graph
```

```python
from dress import fit

# Are these two graphs the same structure?
# Prism (C3 x K2): 6 vertices, both directions listed explicitly
prism = fit(
    6,
    [0,1,1,2,2,0,0,3,1,4,2,5,3,4,4,5,5,3],
    [1,0,2,1,0,2,3,0,4,1,5,2,4,3,5,4,3,5],
)

# K3,3: bipartite {0,1,2} <-> {3,4,5}, again with both directions
k33 = fit(
    6,
    [0,3,0,4,0,5,1,3,1,4,1,5,2,3,2,4,2,5],
    [3,0,4,0,5,0,3,1,4,1,5,1,3,2,4,2,5,2],
)

print("Prism:", sorted(prism.edge_dress))
print("K3,3: ", sorted(k33.edge_dress))
print("Distinguished:", sorted(prism.edge_dress) != sorted(k33.edge_dress))
# → True (DRESS separates them; 1-WL cannot)
```

## Δ^k-DRESS (higher-order refinement)

```python
from dress import delta_fit

result = delta_fit(
    n_vertices=4,
    sources=[0, 1, 2, 0],
    targets=[1, 2, 3, 3],
    k=1,              # delete 1 vertex at a time
    epsilon=1e-6,
)
print(result.histogram)   # exact histogram entries: [(value, count), ...]
print(result.num_subgraphs)
```

## Interactive notebooks

| Notebook | Description |
|----------|-------------|
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/velicast/dress-graph/blob/main/notebooks/quickstart.ipynb) | Quickstart: DRESS on the Karate Club graph |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/velicast/dress-graph/blob/main/notebooks/prism_vs_k33.ipynb) | Prism vs K₃,₃: separates graphs that 1-WL cannot |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/velicast/dress-graph/blob/main/notebooks/delta_dress_rook_shrikhande.ipynb) | Rook vs Shrikhande: Δ¹-DRESS beats 3-WL |

## Why DRESS?

| | Δᵏ-DRESS | WL (color refinement) | GNNs |
|---|---|---|---|
| **Output** | Continuous edge vector | Discrete color histogram | Learned embedding |
| **Parameters** | Zero | Zero | Millions |
| **Training data** | None | None | Required |
| **Per-iteration cost** | O(C(n,k) · m · d_max) | O(n^(k+1)) for k-WL | Varies |
| **Deterministic** | Yes | Yes | No |
| **Expressiveness** | Provably ≥ 2-WL, empirically ≥ (k+2)-WL | = k-WL at level k | ≤ 1-WL (standard MPNN) |

## Applications

- **[Graph Isomorphism](https://velicast.github.io/dress-graph/applications/isomorphism/):** Δ¹-DRESS separates **100% of 51,816** hard benchmark graphs (576M+ pairs). Strictly exceeds 3-WL.
- **[Classification](https://velicast.github.io/dress-graph/applications/classification/):** DRESS fingerprints as features for standard classifiers, matching WL baselines on TU datasets.
- **[Community Detection](https://velicast.github.io/dress-graph/applications/community-detection/):** Edge values naturally classify intra- vs inter-community edges.
- **[Graph Retrieval](https://velicast.github.io/dress-graph/applications/retrieval/):** Fingerprint distances correlate with graph edit distance.
- **[GED Regression](https://velicast.github.io/dress-graph/applications/ged-regression/):** 15× lower MSE than TaGSim on LINUX graphs, no GNN needed.
- **[Edge Robustness](https://velicast.github.io/dress-graph/applications/edge-robustness/):** O(km) edge-importance ranking beating betweenness centrality.
- **[DRESS + GNN](https://velicast.github.io/dress-graph/applications/dress-gnns/):** Plug-in features that boost GIN/PNA/GPS on ZINC-12K.

## Benchmarks

Convergence on real-world graphs (ε = 10⁻⁶):

| Graph | Vertices | Edges | Iterations |
|-------|----------|-------|------------|
| Amazon co-purchasing | 548K | 926K | 18 |
| Wiki-Vote | 8K | 104K | 17 |
| LiveJournal | 4M | 28M | 30 |
| Facebook | 59M | 93M | 26 |

Even on graphs with 59M vertices, DRESS converges in fewer than 31 iterations.

## The equation

Fixed-point form:

$$
d_{uv} = \frac{\sum_{x \in N[u] \cap N[v]} \bigl(\bar{w}_{ux} \, d_{ux} + \bar{w}_{vx} \, d_{vx}\bigr)}{\lVert u \rVert \cdot \lVert v \rVert}
$$

Discrete-time iteration:

$$
d_{uv}^{(t+1)} = \frac{\sum_{x \in N[u] \cap N[v]} \bigl(\bar{w}_{ux} \, d_{ux}^{(t)} + \bar{w}_{vx} \, d_{vx}^{(t)}\bigr)}{\lVert u \rVert^{(t)} \cdot \lVert v \rVert^{(t)}}
$$

where the vertex norm is

$$
\lVert u \rVert^{(t)} = \sqrt{\sum_{x \in N[u]} \bar{w}_{ux} \, d_{ux}^{(t)}}
$$

and $N[u] = N(u) \cup \lbrace u \rbrace$ is the closed neighborhood.

<details>
<summary><b>Key properties</b></summary>

| Property |
|----------|
| Edge-centric refinement (operates only on edges) |
| Parameter-free core (no damping factor, no hyperparameters) |
| Deterministic iterative refinement with practical convergence from a fixed initialization |
| Bounded exactly in [0, 2] for unweighted graphs, self-similarity $d_{uu} = 2$ |
| Isomorphism-invariant |
| Guaranteed bitwise-equal results regardless of vertex labeling (sort + KBN compensated summation) |
| Symmetric by design ($d(u,v) = d(v,u)$ for all pairs) |
| Scale-invariant (degree-0 homogeneous) |
| Completely deterministic |
| Practical convergence in few iterations (contraction)|
| Continuous canonical fingerprints (sorted values / exact sparse histogram) |
| Theoretical per-iteration $\mathcal{O}(\|V\| + \|E\|)$, memory $\mathcal{O}(\|V\| + \|E\|)$ |
| Massively parallelizable ($\Delta^k$ subproblems and per-edge updates) |
| Native weighted-graph support via symmetric weight function |
| Supports directed graphs (four variants: undirected, directed, forward, backward) |
| Provably numerically stable (no overflows, no error amplification, no undefined behaviors) |
| Provably at least as powerful as 2-WL (>= 2-WL) |
| [Locally invertible](https://velicast.github.io/dress-graph/theory/properties/#local-invertibility-incremental-edge-query): Any single edge value recoverable from its neighborhood in O(deg) after one global fit |

</details>

<details>
<summary><b>Isomorphism results (SRG & CFI)</b></summary>

**Δ¹-DRESS: 51,816 graphs, 34 hard families, 100 % separated** ([paper](https://github.com/velicast/dress-graph/blob/main/research/delta1-dress-hard-families.pdf))

Plain DRESS (Δ⁰) assigns a single uniform value to every edge in an SRG, producing zero separation. Δ¹-DRESS breaks this symmetry by running DRESS on each vertex-deleted subgraph. Tested on the complete [Spence SRG collection](https://www.maths.gla.ac.uk/~es/srgraphs.php) (12 families, 43,703 graphs on up to 64 vertices), four additional SRG families from [McKay's collections](https://users.cecs.anu.edu.au/~bdm/data/graphs.html) (8,015 graphs), and 18 constructed hard families (102 graphs including Miyazaki, Chang, Paley, Latin square, and Steiner constructions):

| Category | Families | Graphs | Pairs resolved | Separated |
|----------|:--------:|:------:|:--------------:|:---------:|
| Spence SRG collection | 12 | 43,703 | 559,974,510 | **100 %** |
| Additional SRG families | 4 | 8,015 | 16,132,661 | **100 %** |
| Constructed hard families | 18 | 102 | 664 | **100 %** |
| **Total (distinct)** | **34** | **51,816** | **576,107,835** | **100 %** |

Δ¹-DRESS is strictly more powerful than 3-WL: the Rook L₂(4) vs. Shrikhande pair SRG(16,6,2,2), known to defeat 3-WL, is separated.

**CFI Staircase: Δᵏ-DRESS climbs the WL hierarchy**

The [CFI construction](https://en.wikipedia.org/wiki/Cai%E2%80%93F%C3%BCrer%E2%80%93Immerman_graph) produces the canonical hard instances for every WL level. Δᵏ-DRESS matches $(k{+}2)$-WL on each:

| Base graph | \|V(CFI)\| | WL req. | Δ⁰ | Δ¹ | Δ² | Δ³ |
|:----------:|:----------:|:-------:|:--:|:--:|:--:|:--:|
| $K_3$ | 6 | 2-WL | ✓ | ✓ | ✓ | ✓ |
| $K_4$ | 16 | 3-WL | ✗ | ✓ | ✓ | ✓ |
| $K_5$ | 40 | 4-WL | ✗ | ✗ | ✓ | ✓ |
| $K_6$ | 96 | 5-WL | ✗ | ✗ | ✗ | ✓ |
| $K_7$ | 224 | 6-WL | ✗ | ✗ | ✗ | ✗ |

Each deletion level adds exactly one WL dimension. See [Paper 2](https://github.com/velicast/dress-graph/blob/main/research/vertex-k-DRESS.pdf) for proofs and the full table up to $K_{10}$.

**Standard benchmarks: Original-DRESS (Δ⁰)**

| Benchmark | Accuracy |
|-----------|----------|
| MiVIA database | 100 % |
| IsoBench | 100 % |

</details>

## Examples & experiments

End-to-end examples for every language and backend (CPU, CUDA, MPI, MPI+CUDA) live in **[`examples/`](examples/)**.

Reproducible benchmarks, datasets, and experiment scripts (isomorphism, classification, retrieval, GED regression, and more) live in the dedicated **[dress-experiments](https://github.com/velicast/dress-experiments)** repository.

## Language bindings

C · C++ · Python · Rust · Go · Julia · R · MATLAB/Octave · JavaScript/WASM

All backends: **CPU**, **OpenMP**, **CUDA**, **MPI**, **MPI+OpenMP**, **MPI+CUDA**. Switch by changing the import.

<details>
<summary><b>Quick examples (all languages)</b></summary>

<details>
<summary>C</summary>

```c
#include "dress/dress.h"            // CPU
#include "dress/omp/dress.h"        // OpenMP
#include "dress/cuda/dress.h"       // CUDA  (replaces CPU header)
#include "dress/mpi/dress.h"        // MPI   (replaces CPU header)
#include "dress/mpi/omp/dress.h"    // MPI + OpenMP
#include "dress/mpi/cuda/dress.h"   // MPI + CUDA

p_dress_graph_t g = dress_init_graph(N, E, U, V, W, NULL, DRESS_VARIANT_UNDIRECTED, 0);
int hs; int64_t ns;
dress_hist_pair_t *hist = dress_delta_fit(g, /*k=*/1, /*iter=*/100, /*eps=*/1e-6,
                                          &hs, /*keep_multisets=*/0, NULL, &ns);
dress_free_graph(g);
```

</details>

<details>
<summary>C++</summary>

```cpp
#include "dress/dress.hpp"              // CPU           → ::DRESS
#include "dress/omp/dress.hpp"          // OpenMP        → omp::DRESS
#include "dress/cuda/dress.hpp"         // CUDA          → cuda::DRESS
#include "dress/mpi/dress.hpp"          // MPI           → mpi::DRESS
#include "dress/mpi/omp/dress.hpp"      // MPI + OpenMP  → mpi::omp::DRESS
#include "dress/mpi/cuda/dress.hpp"     // MPI + CUDA    → mpi::cuda::DRESS

DRESS g(N, U, V);                       // (or omp::DRESS, cuda::DRESS, …)
auto r = g.deltaFit(/*k=*/1, /*maxIter=*/100, /*eps=*/1e-6);
// r.histogram, r.num_subgraphs
```

</details>

<details>
<summary>Python</summary>

```python
from dress import delta_fit            # CPU
from dress.omp import delta_fit        # OpenMP
from dress.cuda import delta_fit       # CUDA
from dress.mpi import delta_fit        # MPI
from dress.mpi.omp import delta_fit    # MPI + OpenMP
from dress.mpi.cuda import delta_fit   # MPI + CUDA

result = delta_fit(
    n_vertices=6, sources=[0,1,2,0,1,2,3,4,5],
    targets=[1,2,0,3,4,5,4,5,3], k=1,
)
print(result.histogram)  # [(value, count), ...]
```

</details>

<details>
<summary>Rust</summary>

```rust
use fit::DRESS;                     // CPU
use fit::omp::DRESS;                // OpenMP
use fit::cuda::DRESS;               // CUDA
use fit::mpi;                       // MPI
use fit::mpi::omp;                  // MPI + OpenMP
use fit::mpi::cuda;                 // MPI + CUDA

let r = DRESS::delta_fit(
    6, sources, targets, None,
    /*k=*/1, /*max_iter=*/100, /*eps=*/1e-6,
    Variant::Undirected, false, false, 0, 1,
)?;
```

</details>

<details>
<summary>Go</summary>

```go
import dress "github.com/velicast/dress-graph/go"            // CPU
import dress "github.com/velicast/dress-graph/go/omp"         // OpenMP
import dress "github.com/velicast/dress-graph/go/cuda"        // CUDA
import dress "github.com/velicast/dress-graph/go/mpi"         // MPI
import dress "github.com/velicast/dress-graph/go/mpi/omp"     // MPI + OpenMP
import dress "github.com/velicast/dress-graph/go/mpi/cuda"    // MPI + CUDA

r, err := dress.DeltaFit(6, sources, targets, nil,
    1, dress.Undirected, 100, 1e-6, false, false, 0, 1)
```

</details>

<details>
<summary>Julia</summary>

```julia
using DRESS                      # CPU
using DRESS.OMP                  # OpenMP
using DRESS.CUDA                 # CUDA
using DRESS; using DRESS.MPI     # MPI
using DRESS.MPI.OMP              # MPI + OpenMP

r = delta_fit(6, sources, targets; k=1)
# r.histogram, r.num_subgraphs
```

</details>

<details>
<summary>R</summary>

```r
library(dress.graph)

delta_fit(6, sources, targets, k = 1L)         # CPU
omp$delta_fit(6, sources, targets, k = 1L)      # OpenMP
cuda$delta_fit(6, sources, targets, k = 1L)     # CUDA
mpi$delta_fit(6, sources, targets, k = 1L)      # MPI
mpi$omp$delta_fit(6, sources, targets, k = 1L)  # MPI + OpenMP
mpi$cuda$delta_fit(6, sources, targets, k = 1L) # MPI + CUDA
```

</details>

<details>
<summary>MATLAB / Octave</summary>

```matlab
result = delta_fit(6, sources, targets, 'K', 1);
% result.histogram.value, result.histogram.count, result.num_subgraphs

g = mpi.DRESS(6, int32(sources), int32(targets));        % MPI
go = mpi.omp.DRESS(6, int32(sources), int32(targets));   % MPI + OpenMP
gc = mpi.cuda.DRESS(6, int32(sources), int32(targets));  % MPI + CUDA
```

</details>

<details>
<summary>JavaScript / WASM</summary>

```javascript
import { deltaFit } from './dress.js';

const r = await deltaFit({
    numVertices: 6, sources, targets, k: 1,
});
// r.histogram, r.numSubgraphs
```

</details>

</details>

<details>
<summary><b>Full API reference (all languages × backends)</b></summary>

End-to-end examples for every language × backend live in [`examples/`](examples/).
Each example compares **Prism vs K₃,₃** (Δ⁰-DRESS, CPU/CUDA) or
**Rook L₂(4) vs Shrikhande** (Δ¹-DRESS, MPI/MPI+CUDA, Octave, WASM).

### C

| Backend | Header | Link function | Example |
|---------|--------|---------------|---------|
| CPU | `dress/dress.h` | `-ldress -lm` | [`examples/c/cpu.c`](examples/c/cpu.c) |
| OpenMP | `dress/omp/dress.h` | `-ldress -lm -fopenmp` | [`examples/c/omp.c`](examples/c/omp.c) |
| CUDA | `dress/cuda/dress.h` | `-ldress_cuda -lcudart -lm` | [`examples/c/cuda.c`](examples/c/cuda.c) |
| MPI | `dress/mpi/dress.h` | `mpicc -ldress -lm` | [`examples/c/mpi.c`](examples/c/mpi.c) |
| MPI+OpenMP | `dress/mpi/omp/dress.h` | `mpicc -ldress -lm -fopenmp` | [`examples/c/mpi_omp.c`](examples/c/mpi_omp.c) |
| MPI+CUDA | `dress/mpi/cuda/dress.h` | `mpicc -ldress_cuda -lcudart -lm` | [`examples/c/mpi_cuda.c`](examples/c/mpi_cuda.c) |
| igraph CPU | `dress/igraph/dress.h` | `-ldress $(pkg-config --libs igraph) -lm` | [`examples/c/cpu_igraph.c`](examples/c/cpu_igraph.c) |
| igraph OpenMP | `dress/omp/igraph/dress.h` | `-ldress $(pkg-config --libs igraph) -lm -fopenmp` | [`examples/c/omp_igraph.c`](examples/c/omp_igraph.c) |
| igraph CUDA | `dress/cuda/igraph/dress.h` | `-ldress -ldress_cuda -lcudart $(pkg-config --libs igraph) -lm` | [`examples/c/cuda_igraph.c`](examples/c/cuda_igraph.c) |
| igraph MPI | `dress/mpi/igraph/dress.h` | `mpicc -ldress $(pkg-config --libs igraph) -lm` | [`examples/c/mpi_igraph.c`](examples/c/mpi_igraph.c) |
| igraph MPI+OpenMP | `dress/mpi/omp/igraph/dress.h` | `mpicc -ldress $(pkg-config --libs igraph) -lm -fopenmp` | [`examples/c/mpi_omp_igraph.c`](examples/c/mpi_omp_igraph.c) |
| igraph MPI+CUDA | `dress/mpi/cuda/igraph/dress.h` | `mpicc -ldress -ldress_cuda -lcudart $(pkg-config --libs igraph) -lm` | [`examples/c/mpi_cuda_igraph.c`](examples/c/mpi_cuda_igraph.c) |

```c
// Δ⁰ : edge fingerprint
p_dress_graph_t g = dress_init_graph(N, E, U, V, NULL, NULL, DRESS_VARIANT_UNDIRECTED, 0);
int iters; double delta;
dress_fit(g, 100, 1e-6, &iters, &delta);        // CPU
dress_fit_omp(g, 100, 1e-6, &iters, &delta);    // OpenMP
dress_fit_cuda(g, 100, 1e-6, &iters, &delta);   // CUDA

// Δ¹ : histogram fingerprint
int hs; int64_t ns;
dress_hist_pair_t *hist = dress_delta_fit(g, /*k=*/1, 100, 1e-6, &hs, 0, NULL, &ns);
free(hist);

dress_hist_pair_t *hist_omp = dress_delta_fit_omp(g, 1, 100, 1e-6, &hs, 0, NULL, &ns);
free(hist_omp);

dress_hist_pair_t *hist_mpi = dress_delta_fit_mpi(g, 1, 100, 1e-6, &hs, 0, NULL, &ns, MPI_COMM_WORLD);
free(hist_mpi);

dress_hist_pair_t *hist_mpi_omp = dress_delta_fit_mpi_omp(g, 1, 100, 1e-6, &hs, 0, NULL, &ns, MPI_COMM_WORLD);
free(hist_mpi_omp);

dress_hist_pair_t *hist_mpi_cuda = dress_delta_fit_mpi_cuda(g, 1, 100, 1e-6, &hs, 0, NULL, &ns, MPI_COMM_WORLD);
free(hist_mpi_cuda);

dress_free_graph(g);

// igraph wrapper : same API names, transparent backend switching
#include <dress/igraph/dress.h>           // CPU
#include <dress/cuda/igraph/dress.h>      // CUDA
#include <dress/mpi/igraph/dress.h>       // MPI
#include <dress/mpi/cuda/igraph/dress.h>  // MPI + CUDA

dress_result_igraph_t r;
dress_fit(&graph, NULL, NULL, DRESS_VARIANT_UNDIRECTED,
          100, 1e-6, 1, &r);             // same call for all backends
dress_free(&r);

delta_dress_result_igraph_t dr;
dress_delta_fit(&graph, NULL, NULL, DRESS_VARIANT_UNDIRECTED,
                /*k=*/1, 100, 1e-6, 1, &dr);  // CPU / MPI / CUDA, per header
delta_dress_free(&dr);
```

### C++

| Backend | Header | Namespace | Example |
|---------|--------|-----------|---------|
| CPU | `dress/dress.hpp` | `::DRESS` | [`examples/cpp/cpu.cpp`](examples/cpp/cpu.cpp) |
| OpenMP | `dress/omp/dress.hpp` | `omp::DRESS` | [`examples/cpp/omp.cpp`](examples/cpp/omp.cpp) |
| CUDA | `dress/cuda/dress.hpp` | `cuda::DRESS` | [`examples/cpp/cuda.cpp`](examples/cpp/cuda.cpp) |
| MPI | `dress/mpi/dress.hpp` | `mpi::DRESS` | [`examples/cpp/mpi.cpp`](examples/cpp/mpi.cpp) |
| MPI+OpenMP | `dress/mpi/omp/dress.hpp` | `mpi::omp::DRESS` | [`examples/cpp/mpi_omp.cpp`](examples/cpp/mpi_omp.cpp) |
| MPI+CUDA | `dress/mpi/cuda/dress.hpp` | `mpi::cuda::DRESS` | [`examples/cpp/mpi_cuda.cpp`](examples/cpp/mpi_cuda.cpp) |

```cpp
DRESS g(N, sources, targets);              // or cuda::DRESS, mpi::DRESS, mpi::cuda::DRESS
auto r = g.fit(100, 1e-6);                 // → {iterations, delta}
auto d = g.deltaFit(/*k=*/1);             // → {histogram, num_subgraphs}
double val = g.edgeDress(e);               // per-edge value after fit
```

### Python

| Backend | Import | Example |
|---------|--------|---------|
| CPU | `from dress import fit, delta_fit` | [`cpu.py`](examples/python/cpu.py) |
| OpenMP | `from dress.omp import fit, delta_fit` | [`omp.py`](examples/python/omp.py) |
| CUDA | `from dress.cuda import fit, delta_fit` | [`cuda.py`](examples/python/cuda.py) |
| MPI | `from dress.mpi import delta_fit` | [`mpi.py`](examples/python/mpi.py) |
| MPI+OpenMP | `from dress.mpi.omp import delta_fit` | [`mpi_omp.py`](examples/python/mpi_omp.py) |
| MPI+CUDA | `from dress.mpi.cuda import delta_fit` | [`mpi_cuda.py`](examples/python/mpi_cuda.py) |
| NetworkX (CPU) | `from dress.networkx import fit, delta_fit` | [`cpu_nx.py`](examples/python/cpu_nx.py) |
| NetworkX (OpenMP) | `from dress.omp.networkx import fit, delta_fit` | [`omp_nx.py`](examples/python/omp_nx.py) |
| NetworkX (CUDA) | `from dress.cuda.networkx import fit, delta_fit` | [`cuda_nx.py`](examples/python/cuda_nx.py) |
| NetworkX (MPI) | `from dress.mpi.networkx import delta_fit` | [`mpi_nx.py`](examples/python/mpi_nx.py) |
| NetworkX (MPI+CUDA) | `from dress.mpi.cuda.networkx import delta_fit` | [`mpi_cuda_nx.py`](examples/python/mpi_cuda_nx.py) |
| CPU (OO) | `from dress import DRESS` | [`cpu_oo.py`](examples/python/cpu_oo.py) |
| OpenMP (OO) | `from dress.omp import DRESS` | [`omp_oo.py`](examples/python/omp_oo.py) |
| CUDA (OO) | `from dress.cuda import DRESS` | [`cuda_oo.py`](examples/python/cuda_oo.py) |
| MPI (OO) | `from dress.mpi import DRESS` | [`mpi_oo.py`](examples/python/mpi_oo.py) |
| MPI+OpenMP (OO) | `from dress.mpi.omp import DRESS` | [`mpi_omp_oo.py`](examples/python/mpi_omp_oo.py) |
| MPI+CUDA (OO) | `from dress.mpi.cuda import DRESS` | [`mpi_cuda_oo.py`](examples/python/mpi_cuda_oo.py) |

```python
# Δ⁰ : edge fingerprint
result = fit(n_vertices, sources, targets)
result.edge_dress    # per-edge values
result.vertex_dress    # per-vertex norms
result.iterations    # convergence iterations

# Δ¹ : histogram fingerprint
result = delta_fit(n_vertices, sources, targets, k=1)
result.histogram     # exact histogram entries: [(value, count), ...]

# NetworkX: pass a graph directly
import networkx as nx
from dress.networkx import fit, delta_fit

G = nx.karate_club_graph()
result = fit(G, set_attributes=True)
G.edges[0, 1]["dress"]      # per-edge similarity

delta = delta_fit(G, k=1, keep_multisets=True)
delta.histogram              # exact histogram entries: [(value, count), ...]

# MPI NetworkX: same API, distributed
from dress.mpi.networkx import delta_fit
delta = delta_fit(G, k=1)  # uses MPI.COMM_WORLD
```

### Rust

| Backend | Import | Example |
|---------|--------|---------|
| CPU | `use fit::DRESS` | [`examples/rust/cpu.rs`](examples/rust/cpu.rs) |
| OpenMP | `use fit::omp::DRESS` | [`examples/rust/omp.rs`](examples/rust/omp.rs) |
| CUDA | `use fit::cuda::DRESS` | [`examples/rust/cuda.rs`](examples/rust/cuda.rs) |
| MPI | `use fit::mpi` | [`examples/rust/mpi.rs`](examples/rust/mpi.rs) |
| MPI+OpenMP | `use fit::mpi::omp` | [`examples/rust/mpi_omp.rs`](examples/rust/mpi_omp.rs) |
| MPI+CUDA | `use fit::mpi::cuda` | [`examples/rust/mpi_cuda.rs`](examples/rust/mpi_cuda.rs) |
| CPU (OO) | `use fit::{DRESS, Variant}` | [`examples/rust/cpu_oo.rs`](examples/rust/cpu_oo.rs) |
| CUDA (OO) | `use fit::{cuda, Variant}` | [`examples/rust/cuda_oo.rs`](examples/rust/cuda_oo.rs) |
| MPI (OO) | `use fit::{mpi, Variant}` | [`examples/rust/mpi_oo.rs`](examples/rust/mpi_oo.rs) |
| MPI+CUDA (OO) | `use fit::{mpi, Variant}` | [`examples/rust/mpi_cuda_oo.rs`](examples/rust/mpi_cuda_oo.rs) |

```rust
// Builder pattern (CPU / CUDA)
let result = DRESS::builder(n, sources, targets)
    .variant(Variant::Undirected)
    .build_and_fit()?;
// result.edge_dress, result.iterations, result.delta

// MPI delta fit
let r = mpi::delta_fit(n, sources, targets, None,
    /*k=*/1, 100, 1e-6, Variant::Undirected, false, false, &world)?;
// r.histogram, r.num_subgraphs
```

### Go

| Backend | Import path | Example |
|---------|-------------|---------|
| CPU | `github.com/velicast/dress-graph/go` | [`examples/go/cpu.go`](examples/go/cpu.go) |
| OpenMP | `github.com/velicast/dress-graph/go/omp` | [`examples/go/omp.go`](examples/go/omp.go) |
| CUDA | `github.com/velicast/dress-graph/go/cuda` | [`examples/go/cuda.go`](examples/go/cuda.go) |
| MPI | `github.com/velicast/dress-graph/go/mpi` | [`examples/go/mpi.go`](examples/go/mpi.go) |
| MPI+OpenMP | `github.com/velicast/dress-graph/go/mpi/omp` | [`examples/go/mpi_omp.go`](examples/go/mpi_omp.go) |
| MPI+CUDA | `github.com/velicast/dress-graph/go/mpi/cuda` | [`examples/go/mpi_cuda.go`](examples/go/mpi_cuda.go) |
| CPU (OO) | `github.com/velicast/dress-graph/go` | [`examples/go/cpu_oo.go`](examples/go/cpu_oo.go) |
| CUDA (OO) | `github.com/velicast/dress-graph/go/cuda` | [`examples/go/cuda_oo.go`](examples/go/cuda_oo.go) |
| MPI (OO) | `github.com/velicast/dress-graph/go/mpi` | [`examples/go/mpi_oo.go`](examples/go/mpi_oo.go) |
| MPI+CUDA (OO) | `github.com/velicast/dress-graph/go/mpi/cuda` | [`examples/go/mpi_cuda_oo.go`](examples/go/mpi_cuda_oo.go) |

```go
// CPU / CUDA: same function, different import
result, _ := dress.Fit(n, sources, targets, nil,
    dress.Undirected, 100, 1e-6, false)
// result.EdgeDress, result.Iterations, result.Delta

// MPI / MPI+CUDA: delta only
dress.Init()
defer dress.Finalize()
r, _ := dress.DeltaFit(n, sources, targets, nil,
    /*k=*/1, dress.Undirected, 100, 1e-6, false, false)
// r.Histogram, r.NumSubgraphs
```

### Julia

| Backend | Import | Example |
|---------|--------|---------|
| CPU | `using DRESS` | [`examples/julia/cpu.jl`](examples/julia/cpu.jl) |
| OpenMP | `using DRESS.OMP` | [`examples/julia/omp.jl`](examples/julia/omp.jl) |
| CUDA | `using DRESS.CUDA` | [`examples/julia/cuda.jl`](examples/julia/cuda.jl) |
| MPI | `using DRESS.MPI` | [`examples/julia/mpi.jl`](examples/julia/mpi.jl) |
| MPI+OpenMP | `using DRESS.MPI.OMP` | [`examples/julia/mpi_omp.jl`](examples/julia/mpi_omp.jl) |
| MPI+CUDA | `using DRESS.MPI.CUDA` | [`examples/julia/mpi_cuda.jl`](examples/julia/mpi_cuda.jl) |
| CPU (OO) | `using DRESS` | [`examples/julia/cpu_oo.jl`](examples/julia/cpu_oo.jl) |
| CUDA (OO) | `using DRESS.CUDA` | [`examples/julia/cuda_oo.jl`](examples/julia/cuda_oo.jl) |
| MPI (OO) | `using DRESS.MPI` | [`examples/julia/mpi_oo.jl`](examples/julia/mpi_oo.jl) |
| MPI+CUDA (OO) | `using DRESS.MPI.CUDA` | [`examples/julia/mpi_cuda_oo.jl`](examples/julia/mpi_cuda_oo.jl) |

```julia
# Δ⁰ : edge fingerprint
r = fit(N, sources, targets)
# r.edge_dress, r.vertex_dress, r.iterations, r.delta

# Δ¹ : histogram fingerprint
r = delta_fit(N, sources, targets; k=1)
# r.histogram, r.num_subgraphs
```

### R

| Backend | Call | Example |
|---------|------|---------|
| CPU | `fit()` / `delta_fit()` | [`examples/r/cpu.R`](examples/r/cpu.R) |
| OpenMP | `omp$fit()` / `omp$delta_fit()` | [`examples/r/omp.R`](examples/r/omp.R) |
| CUDA | `cuda$fit()` / `cuda$delta_fit()` | [`examples/r/cuda.R`](examples/r/cuda.R) |
| MPI | `mpi$delta_fit()` | [`examples/r/mpi.R`](examples/r/mpi.R) |
| MPI+OpenMP | `mpi$omp$delta_fit()` | [`examples/r/mpi_omp.R`](examples/r/mpi_omp.R) |
| MPI+CUDA | `mpi$cuda$delta_fit()` | [`examples/r/mpi_cuda.R`](examples/r/mpi_cuda.R) |
| CPU (OO) | `DRESS(...)$fit()` | [`examples/r/cpu_oo.R`](examples/r/cpu_oo.R) |
| CUDA (OO) | `cuda$DRESS(...)$fit()` | [`examples/r/cuda_oo.R`](examples/r/cuda_oo.R) |
| MPI (OO) | `mpi$DRESS(...)$delta_fit()` | [`examples/r/mpi_oo.R`](examples/r/mpi_oo.R) |
| MPI+CUDA (OO) | `mpi$cuda$DRESS(...)$delta_fit()` | [`examples/r/mpi_cuda_oo.R`](examples/r/mpi_cuda_oo.R) |

```r
library(dress.graph)

# Δ⁰ : edge fingerprint
r <- fit(6L, sources, targets)
# r$edge_dress, r$vertex_dress, r$iterations, r$delta

# Δ¹ : histogram fingerprint
r <- delta_fit(6L, sources, targets, k = 1L)       # CPU
r <- omp$delta_fit(6L, sources, targets, k = 1L)    # OpenMP
r <- cuda$delta_fit(6L, sources, targets, k = 1L)   # CUDA
r <- mpi$delta_fit(6L, sources, targets, k = 1L)    # MPI
r <- mpi$omp$delta_fit(6L, sources, targets, k = 1L) # MPI+OpenMP
r <- mpi$cuda$delta_fit(6L, sources, targets, k = 1L) # MPI+CUDA
```

### MATLAB / Octave

| Backend | Function | Example |
|---------|----------|---------|
| CPU | `fit()` / `delta_fit()` | [`examples/octave/cpu.m`](examples/octave/cpu.m) |
| CUDA | `cuda.fit()` | [`examples/octave/cuda_example.m`](examples/octave/cuda_example.m) |
| Δ¹-DRESS | `delta_fit(..., 'K', 1)` | [`examples/octave/rook_vs_shrikhande.m`](examples/octave/rook_vs_shrikhande.m) |
| CPU (OO, Octave) | `DRESS(...)` | [`examples/octave/cpu_oo.m`](examples/octave/cpu_oo.m) |
| OpenMP | `omp.fit()` / `omp.delta_fit()` | [`examples/matlab/omp_oo.m`](examples/matlab/omp_oo.m) |
| MPI (OO, Octave) | `mpi.DRESS(...)` | [`examples/octave/mpi_oo.m`](examples/octave/mpi_oo.m) |
| MPI+OpenMP (OO) | `mpi.omp.DRESS(...)` | [`examples/matlab/mpi_omp_oo.m`](examples/matlab/mpi_omp_oo.m) |
| MPI+CUDA (OO, Octave) | `mpi.cuda.DRESS(...)` | [`examples/octave/mpi_cuda_oo.m`](examples/octave/mpi_cuda_oo.m) |
| CPU (OO, Matlab) | `DRESS(...)` | [`examples/matlab/cpu_oo.m`](examples/matlab/cpu_oo.m) |
| CUDA (OO, Matlab) | `cuda.DRESS(...)` | [`examples/matlab/cuda_oo.m`](examples/matlab/cuda_oo.m) |
| MPI (OO, Matlab) | `mpi.DRESS(...)` | [`examples/matlab/mpi_oo.m`](examples/matlab/mpi_oo.m) |
| MPI+CUDA (OO, Matlab) | `mpi.cuda.DRESS(...)` | [`examples/matlab/mpi_cuda_oo.m`](examples/matlab/mpi_cuda_oo.m) |

```matlab
% Δ⁰ : edge fingerprint
result = fit(6, int32(sources), int32(targets));
% result.edge_dress, result.vertex_dress, result.iterations, result.delta

% Δ¹ : histogram fingerprint
result = delta_fit(6, int32(sources), int32(targets), ...
    'K', 1, 'KeepMultisets', true);
% result.histogram.value, result.histogram.count, result.multisets, result.num_subgraphs

% Persistent graph with get()
g = DRESS(6, int32(sources), int32(targets));
g.fit('MaxIterations', 100, 'Epsilon', 1e-6);
d = g.get(u, v);     % query edge similarity
g.close();

% Octave OO MPI variants
g = mpi.DRESS(6, int32(sources), int32(targets));
r = g.delta_fit('K', 1, 'KeepMultisets', true);
g.close();

g = mpi.cuda.DRESS(6, int32(sources), int32(targets));
r = g.delta_fit('K', 1, 'KeepMultisets', true);
g.close();
```

### JavaScript / WASM

| Backend | Import | Example |
|---------|--------|---------|
| CPU | `import { fit } from './dress.js'` | [`examples/wasm/cpu.mjs`](examples/wasm/cpu.mjs) |
| Δ¹-DRESS | `import { deltaFit } from './dress.js'` | [`examples/wasm/rook_vs_shrikhande.mjs`](examples/wasm/rook_vs_shrikhande.mjs) |
| CPU (OO) | `import { DRESS } from 'dress-graph'` | [`examples/wasm/cpu_oo.mjs`](examples/wasm/cpu_oo.mjs) |

```javascript
// Δ⁰ : edge fingerprint
const result = await fit({
    numVertices: 6, sources, targets,
});
// result.edgeDress, result.vertexDress, result.iterations, result.delta

// Δ¹ : histogram fingerprint
const r = await deltaFit({
    numVertices: 6, sources, targets,
    k: 1, keepMultisets: true,
});
// r.histogram, r.multisets, r.numSubgraphs

// Persistent graph with get()
const g = await DRESS.create({ numVertices: 6, sources, targets });
g.fit(100, 1e-6);
const d = g.get(u, v);   // query edge similarity
g.free();
```

</details>

See the [full documentation →](https://velicast.github.io/dress-graph/) for complete API reference.

## Building from source

```bash
./build.sh --no-test

# Python
cd python && pip install .
```

## Documentation

Full documentation (theory, applications, API reference):
[https://velicast.github.io/dress-graph/](https://velicast.github.io/dress-graph/)

## Publications

- E. Castrillo. *DRESS: A Continuous Framework for Structural Graph Refinement.* [arXiv:2602.20833](https://github.com/velicast/dress-graph/blob/main/research/k-DRESS.pdf)
- E. Castrillo. *Breaking Hard Isomorphism Benchmarks with DRESS.* [arXiv:2603.18582](https://arxiv.org/abs/2603.18582)
- E. Castrillo. *DRESS and the WL Hierarchy: Climbing One Deletion at a Time.* [arXiv:2602.21557](https://arxiv.org/abs/2602.21557)
- E. Castrillo, E. León, J. Gómez. *Dynamic Structural Similarity on Graphs.* [arXiv:1805.01419](https://arxiv.org/abs/1805.01419)
- E. Castrillo, E. León, J. Gómez. *Fast Heuristic Algorithm for Multi-Scale Hierarchical Community Detection.* [ASONAM 2017](https://dl.acm.org/citation.cfm?doid=3110025.3110125)
- E. Castrillo, E. León, J. Gómez. *High-Quality Disjoint and Overlapping Community Structure in Large-Scale Complex Networks.* [arXiv:1805.12238](https://arxiv.org/abs/1805.12238)

Original implementation (2018): [github.com/velicast/WMW](https://github.com/velicast/WMW)

## Citing

If you use DRESS in your research, please cite:

```bibtex
@misc{castrillo2026dress,
  title   = {DRESS: A Continuous Framework for Structural Graph Refinement},
  author  = {Eduar Castrillo Velilla},
  year    = {2026},
  eprint  = {2602.20833},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DS},
  url     = {https://arxiv.org/abs/2602.20833}
}

@misc{castrillo2018dress,
  title   = {Dynamic Structural Similarity on Graphs},
  author  = {Eduar Castrillo and Elizabeth Le{\'o}n and Jonatan G{\'o}mez},
  year    = {2018},
  eprint  = {1805.01419},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SI},
  url     = {https://arxiv.org/abs/1805.01419}
}
```

> **Why "DRESS"?** DRESS computes an edge labeling that reveals the graph's hidden
> structural identity; it *dresses* the bare skeleton (adjacency) with
> meaningful values.  A graph without DRESS is "naked" topology; after DRESS,
> every edge wears the structural role that fits it best.  And `fit()`
> is literally fitting the dress to the graph: few iterations give a loose fit,
> more iterations tighten it, and at steady state the fit is true to size.

## License

[MIT](LICENSE)
