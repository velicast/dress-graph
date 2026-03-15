# dress-graph

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

PyPi [![PyPI Downloads](https://static.pepy.tech/personalized-badge/dress-graph?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/dress-graph) Crates [![crates.io downloads](https://img.shields.io/crates/d/dress-graph)](https://crates.io/crates/dress-graph) NPM [![npm downloads](https://img.shields.io/npm/dt/dress-graph)](https://www.npmjs.com/package/dress-graph) CRAN [![CRAN downloads](https://cranlogs.r-pkg.org/badges/grand-total/dress.graph)](https://cran.r-project.org/package=dress.graph)

Quickstart [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/velicast/dress-graph/blob/main/notebooks/quickstart.ipynb)
Prism vs K₃,₃ [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/velicast/dress-graph/blob/main/notebooks/prism_vs_k33.ipynb)
Rook vs Shrikhande [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/velicast/dress-graph/blob/main/notebooks/delta_dress_rook_shrikhande.ipynb)

## A Continuous Framework for Structural Graph Refinement

We introduce DRESS, a deterministic, parameter-free framework that iteratively refines the structural similarity of edges in a graph to produce a *canonical fingerprint*: a real-valued edge vector, obtained by converging a non-linear dynamical system to its unique fixed point. The fingerprint is *isomorphism-invariant* by construction, *numerically stable* (all values lie in [0, 2]), *fast* and *embarrassingly parallel* to compute: each iteration costs O(m · d_max) and convergence is guaranteed by Birkhoff contraction. As a direct consequence of these properties, DRESS is provably at least as expressive as the 2-dimensional Weisfeiler–Leman (2-WL) test, at a fraction of the cost (O(m · d_max) vs. O(n³) per iteration). We generalize the original equation (Castrillo, León, and Gómez, 2018) to Motif-DRESS (arbitrary structural motifs) and Generalized-DRESS (abstract aggregation template), and introduce Δ-DRESS, which runs DRESS on each vertex-deleted subgraph to boost expressiveness. Δ¹-DRESS empirically separates all 51,816 graphs across 34 hard benchmark families (16 SRG families totaling 51,718 graphs from the complete Spence collection and McKay's additional SRG data, plus 18 constructed hard families), resolving over 576 million within-family non-isomorphic pairs, and is the cheapest known method that strictly exceeds 3-WL. Iterated deletion (Δᵏ-DRESS) climbs the CFI staircase, achieving (k+2)-WL expressiveness at each depth k. The algorithm is embarrassingly parallel in two orthogonal ways - across the vertex-deleted subgraphs and across edge updates within each iteration - enabling distributed/cloud plus multi-core/GPU/SIMD implementations. Successfully applied to a handful of downstream applications.

> **Note on Wrappers:** Please report any bugs you find while using the language wrappers (Python, Rust, JS, etc.). I am moving quickly and relying on AI to speed up the development of the wrappers, but I am directly and carefully maintaining the core C backend.

The arXiv papers is outdated and will be updated next week.

For the theory and generalizations (DRESS Family), see the research paper:
[**arXiv:2602.20833**](https://github.com/velicast/dress-graph/blob/main/research/k-DRESS.pdf)

## The equation

Fixed-point form:

$$
d_{uv} = \frac{\displaystyle\sum_{x \in N[u] \cap N[v]}
  \bigl(\bar{w}_{ux}\, d_{ux} + \bar{w}_{vx}\, d_{vx}\bigr)}
  {\|u\| \cdot \|v\|}
$$

Discrete-time iteration:

$$
d_{uv}^{(t+1)} = \frac{\displaystyle\sum_{x \in N[u] \cap N[v]}
  \bigl(\bar{w}_{ux}\, d_{ux}^{(t)} + \bar{w}_{vx}\, d_{vx}^{(t)}\bigr)}
  {\|u\|^{(t)} \cdot \|v\|^{(t)}}
$$

where the node norm is

$$
\|u\|^{(t)} = \sqrt{\sum_{x \in N[u]} \bar{w}_{ux}\, d_{ux}^{(t)}}
$$

and $N[u] = N(u) \cup \\{u\\}$ is the closed neighborhood.

## Key properties

| Property |
|----------|
| Edge-centric refinement (operates only on edges) |
| Parameter-free core (no damping factor, no hyperparameters) |
| Unique fixed point via Birkhoff contraction |
| Bounded exactly in [0, 2] for unweighted graphs, self-similarity $d_{uu} = 2$ |
| Isomorphism-invariant |
| Symmetric by design ($d(u,v) = d(v,u)$ for all pairs) |
| Scale-invariant (degree-0 homogeneous) |
| Completely deterministic |
| Practical convergence in few iterations (contraction)|
| Continuous canonical fingerprints (sorted values / ε-binned histogram) |
| Theoretical per-iteration $\mathcal{O}(\|V\| + \|E\|)$, memory $\mathcal{O}(\|V\| + \|E\|)$ |
| Massively parallelizable ($\Delta^k$ subproblems and per-edge updates) |
| Native weighted-graph support via symmetric weight function |
| Supports directed graphs (four variants: undirected, directed, forward, backward) |
| Provably numerically stable (no overflows, no error amplification, no undefined behaviors) |
| Provably at least as powerful as 2-WL (>= 2-WL) |
| [Locally invertible](https://velicast.github.io/dress-graph/theory/properties/#local-invertibility-incremental-edge-query): Any single edge value recoverable from its neighborhood in O(deg) after one global fit |

## Benchmarks

Convergence on real-world graphs (tolerance ε = 10⁻⁶, max 100 iterations):

| Graph | Vertices | Edges | Iterations | Final δ |
|-------|----------|-------|------------|---------|
| Amazon product co-purchasing | 548,552 | 925,872 | 18 | 6.35e-7 |
| Wiki-Vote | 8,298 | 103,689 | 17 | 8.31e-7 |
| LiveJournal social network | 4,033,138 | 27,933,062 | 30 | 7.09e-7 |
| Facebook (konect) | 59,216,215 | 92,522,012 | 26 | 6.84e-7 |
| Facebook (UCI/UNI) | 58,790,783 | 92,208,195 | 26 | 6.84e-7 |

- **Low iteration count.** Even on graphs with tens of millions of vertices and edges, DRESS converges in fewer than 31 iterations - consistent with the contraction-mapping guarantee.
- **Scale independence.** Iteration count grows very slowly with graph size. A graph with 59 M vertices needs only ~1.5× the iterations of one with 8 K vertices.
- **Uniform residual.** The final δ is consistently on the order of 10⁻⁷, indicating that convergence quality does not degrade with graph size.

## Current Experimented Applications

- **[Graph Isomorphism](https://velicast.github.io/dress-graph/applications/isomorphism/)**: sorting DRESS edge values produces a canonical fingerprint.

  **Δ¹-DRESS: 51,816 graphs, 34 hard families, 100 % separated** ([paper](https://github.com/velicast/dress-graph/blob/main/research/delta1-dress-hard-families.pdf))

  Plain DRESS (Δ⁰) assigns a single uniform value to every edge in an SRG, producing zero separation. Δ¹-DRESS breaks this symmetry by running DRESS on each vertex-deleted subgraph. Tested on the complete [Spence SRG collection](https://www.maths.gla.ac.uk/~es/srgraphs.php) (12 families, 43,703 graphs on up to 64 vertices), four additional SRG families from [McKay's collections](https://users.cecs.anu.edu.au/~bdm/data/graphs.html) (8,015 graphs), and 18 constructed hard families (102 graphs including Miyazaki, Chang, Paley, Latin square, and Steiner constructions):

  | Category | Families | Graphs | Pairs resolved | Separated |
  |----------|:--------:|:------:|:--------------:|:---------:|
  | Spence SRG collection | 12 | 43,703 | 559,974,510 | **100 %** |
  | Additional SRG families | 4 | 8,015 | 16,132,661 | **100 %** |
  | Constructed hard families | 18 | 102 | 664 | **100 %** |
  | **Total (distinct)** | **34** | **51,816** | **576,107,835** | **100 %** |

  Δ¹-DRESS is strictly more powerful than 3-WL: the Rook L₂(4) vs. Shrikhande pair SRG(16,6,2,2), known to defeat 3-WL, is separated. This places Δ¹-DRESS strictly above 3-WL; whether it is bounded above by 4-WL (≡ 3-FWL) remains open.

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
- **[Community Detection](https://velicast.github.io/dress-graph/applications/community-detection/)**: DRESS values classify edges as intra- or inter-community, improving SCAN and enabling agglomerative hierarchical clustering.
- **[Classification](https://velicast.github.io/dress-graph/applications/classification/)**: percentile-based DRESS fingerprints fed to standard classifiers match or exceed Weisfeiler-Leman baselines on TU benchmark datasets.
- **[Retrieval](https://velicast.github.io/dress-graph/applications/retrieval/)**: DRESS fingerprint distances correlate strongly with graph edit distance, achieving state-of-the-art precision on GED-based retrieval benchmarks.
- **[GED Regression](https://velicast.github.io/dress-graph/applications/ged-regression/)**: DRESS fingerprint differences fed to a simple regressor predict graph edit distance with 15× lower MSE than TaGSim on LINUX graphs - no GNN required.
- **[Edge Robustness](https://velicast.github.io/dress-graph/applications/edge-robustness/)**: DRESS edge values double as an O(km) edge-importance ranking that outperforms O(nm) betweenness centrality and four other baselines (65–97% win rates, p < 0.0001 across 224 graphs).
- **[DRESS + GNN](https://velicast.github.io/dress-graph/applications/dress-gnns/)**: DRESS node/edge values injected as plug-in features into GIN, PNA, GPS, and custom DRESSNet architectures on ZINC-12K molecular property prediction. GIN+bond+DRESS drops MAE from 0.526 → 0.235; PNA+bond+DRESS achieves 0.212 MAE, competitive with the published PNA baseline (0.188).

Full experimental setups, datasets, and scripts are available in the **[dress-experiments](https://github.com/velicast/dress-experiments)** repository.

## Experiments

All benchmarks, datasets, and reproducible scripts live in a dedicated repository:

**[dress-experiments](https://github.com/velicast/dress-experiments)** - isomorphism (SRG, CFI, constructed families), classification, retrieval, GED regression, and more.

## Quick start (Python)

```bash
pip install dress-graph
```

```python
from dress import dress_fit

result = dress_fit(
    n_vertices=4,
    sources=[0, 1, 2, 0],
    targets=[1, 2, 3, 3],
)
print(result.edge_dress)  # DRESS value for each edge
```

### Δ^k-DRESS (higher-order refinement)

```python
from dress import delta_dress_fit

result = delta_dress_fit(
    n_vertices=4,
    sources=[0, 1, 2, 0],
    targets=[1, 2, 3, 3],
    k=1,              # delete 1 vertex at a time
    epsilon=1e-6,
)
print(result.histogram)   # histogram of edge DRESS values across all subgraphs
print(result.hist_size)   # number of bins
```

```python
#------------------------
# pip install dress-graph
#------------------------

"""Prism vs K3,3 (both 3-regular on 6 nodes). Confound 1-WL."""

from dress import dress_fit

# Prism (C3 □ K2): triangles 0-1-2 and 3-4-5, spokes 0-3 1-4 2-5
prism_s = [0, 1, 2, 0, 1, 2, 3, 4, 5]
prism_t = [1, 2, 0, 3, 4, 5, 4, 5, 3]

# K3,3: bipartite {0,1,2} ↔ {3,4,5}
k33_s = [0, 0, 0, 1, 1, 1, 2, 2, 2]
k33_t = [3, 4, 5, 3, 4, 5, 3, 4, 5]

r1 = dress_fit(6, prism_s, prism_t)
r2 = dress_fit(6, k33_s, k33_t)

print("Prism edge_dress:", sorted(r1.edge_dress))
print("K3,3  edge_dress:", sorted(r2.edge_dress))
fp1 = sorted(round(val, 6) for val in r1.edge_dress)
fp2 = sorted(round(val, 6) for val in r2.edge_dress)
print("Distinguished:", fp1 != fp2)
```

## Interactive notebooks

| Notebook | Description |
|----------|-------------|
| [Quickstart](notebooks/quickstart.ipynb) | Compute DRESS on the Karate Club graph and visualize edge roles |
| [Prism vs K₃,₃](notebooks/prism_vs_k33.ipynb) | DRESS distinguishes two 3-regular graphs that confound 1-WL |
| [Rook vs Shrikhande](notebooks/delta_dress_rook_shrikhande.ipynb) | Δ¹-DRESS distinguishes two cospectral SRG(16,6,2,2) mates |

## Language bindings

DRESS is implemented in C with bindings for:

- **C / C++**: static and shared libraries ([Homebrew](https://github.com/velicast/homebrew-dress-graph), [vcpkg](vcpkg/), or build from source)
- **Python**: pybind11 (`pip install dress-graph`)
- **Rust**: `dress-graph` crate
- **Go**: CGo bindings
- **Julia**: native FFI module
- **R**: `.Call` interface
- **MATLAB / Octave**: MEX gateway
- **JavaScript / WASM**: browser and Node.js (CPU only)

All backends (**CPU**, **CUDA** (GPU), **MPI** (distributed), and **MPI+CUDA**) are supported across all native language bindings.
An **igraph** C wrapper (`libdress-igraph`) is also available with the same CPU / CUDA / MPI / MPI+CUDA backend matrix.
JavaScript / WASM is CPU-only (browser).

Every binding exposes the same `delta_dress_fit` function. Switching between CPU, CUDA,
MPI, and MPI+CUDA is done purely by changing the import / include:

<details>
<summary><b>C</b></summary>

```c
#include "dress/dress.h"            // CPU
#include "dress/cuda/dress.h"       // CUDA  (replaces CPU header)
#include "dress/mpi/dress.h"        // MPI   (replaces CPU header)
#include "dress/mpi/cuda/dress.h"   // MPI + CUDA (single include)

p_dress_graph_t g = init_dress_graph(N, E, U, V, W, DRESS_VARIANT_UNDIRECTED, 0);
int hs; int64_t ns;
int64_t *hist = delta_dress_fit(g, /*k=*/1, /*iter=*/100, /*eps=*/1e-6,
                                &hs, /*keep_multisets=*/0, NULL, &ns);
free_dress_graph(g);
```

</details>

<details>
<summary><b>C++</b></summary>

```cpp
#include "dress/dress.hpp"              // CPU           → ::DRESS
#include "dress/cuda/dress.hpp"         // CUDA          → cuda::DRESS
#include "dress/mpi/dress.hpp"          // MPI           → mpi::DRESS
#include "dress/mpi/cuda/dress.hpp"     // MPI + CUDA    → mpi::cuda::DRESS

DRESS g(N, U, V);                       // (or cuda::DRESS, mpi::DRESS, …)
auto r = g.deltaFit(/*k=*/1, /*maxIter=*/100, /*eps=*/1e-6);
// r.histogram, r.hist_size, r.num_subgraphs
```

</details>

<details>
<summary><b>Python</b></summary>

```python
from dress import delta_dress_fit            # CPU
from dress.cuda import delta_dress_fit       # CUDA
from dress.mpi import delta_dress_fit        # MPI
from dress.mpi.cuda import delta_dress_fit   # MPI + CUDA

result = delta_dress_fit(
    n_vertices=6, sources=[0,1,2,0,1,2,3,4,5],
    targets=[1,2,0,3,4,5,4,5,3], k=1,
)
print(result.histogram, result.hist_size)
```

</details>

<details>
<summary><b>Rust</b></summary>

```rust
use dress_graph::DRESS;                     // CPU
use dress_graph::cuda::DRESS;               // CUDA

// MPI / MPI + CUDA
use dress_graph::mpi;
use dress_graph::mpi::cuda;

let r = DRESS::delta_fit(
    6, sources, targets, None,
    /*k=*/1, /*max_iter=*/100, /*eps=*/1e-6,
    Variant::Undirected, false, false, 0, 1,
)?;
// MPI variant:
let r = mpi::delta_fit(6, sources, targets, None,
    1, 100, 1e-6, Variant::Undirected, false, false, &comm)?;
```

</details>

<details>
<summary><b>Go</b></summary>

```go
import dress "github.com/velicast/dress-graph/go"            // CPU
import dress "github.com/velicast/dress-graph/go/cuda"        // CUDA
import dress "github.com/velicast/dress-graph/go/mpi"         // MPI
import dress "github.com/velicast/dress-graph/go/mpi/cuda"    // MPI + CUDA

r, err := dress.DeltaDressFit(6, sources, targets, nil,
    1, dress.Undirected, 100, 1e-6, false, false, 0, 1)
// r.Histogram, r.HistSize, r.NumSubgraphs
```

</details>

<details>
<summary><b>Julia</b></summary>

```julia
using DRESS                      # CPU
using DRESS.CUDA                 # CUDA
using DRESS; using DRESS.MPI     # MPI
using DRESS; using DRESS.MPI.CUDA  # MPI + CUDA

r = delta_dress_fit(6, sources, targets; k=1)
# r.histogram, r.hist_size, r.num_subgraphs
```

</details>

<details>
<summary><b>R</b></summary>

```r
library(dress.graph)

delta_dress_fit(6, sources, targets, k = 1L)         # CPU
cuda$delta_dress_fit(6, sources, targets, k = 1L)     # CUDA
mpi$delta_dress_fit(6, sources, targets, k = 1L)      # MPI
mpi$cuda$delta_dress_fit(6, sources, targets, k = 1L) # MPI + CUDA
```

</details>

<details>
<summary><b>MATLAB / Octave</b></summary>

```matlab
result = delta_dress_fit(6, sources, targets, 'K', 1);
% result.histogram, result.hist_size, result.num_subgraphs
```

</details>

<details>
<summary><b>JavaScript / WASM</b></summary>

```javascript
import { deltaDressFit } from './dress.js';

const r = await deltaDressFit({
    numVertices: 6, sources, targets, k: 1,
});
// r.histogram, r.histSize, r.numSubgraphs
```

</details>

## API Reference

End-to-end examples for every language × backend live in [`examples/`](examples/).
Each example compares **Prism vs K₃,₃** (Δ⁰-DRESS, CPU/CUDA) or
**Rook L₂(4) vs Shrikhande** (Δ¹-DRESS, MPI/MPI+CUDA, Octave, WASM).

### C

| Backend | Header | Link function | Example |
|---------|--------|---------------|---------|
| CPU | `dress/dress.h` | `-ldress -lm` | [`examples/c/cpu.c`](examples/c/cpu.c) |
| CUDA | `dress/cuda/dress.h` | `-ldress_cuda -lcudart -lm` | [`examples/c/cuda.c`](examples/c/cuda.c) |
| MPI | `dress/mpi/dress.h` | `mpicc -ldress -lm` | [`examples/c/mpi.c`](examples/c/mpi.c) |
| MPI+CUDA | `dress/mpi/cuda/dress.h` | `mpicc -ldress_cuda -lcudart -lm` | [`examples/c/mpi_cuda.c`](examples/c/mpi_cuda.c) |
| igraph CPU | `dress/igraph/dress.h` | `-ldress $(pkg-config --libs igraph) -lm` | [`examples/c/cpu_igraph.c`](examples/c/cpu_igraph.c) |
| igraph CUDA | `dress/cuda/igraph/dress.h` | `-ldress -ldress_cuda -lcudart $(pkg-config --libs igraph) -lm` | [`examples/c/cuda_igraph.c`](examples/c/cuda_igraph.c) |
| igraph MPI | `dress/mpi/igraph/dress.h` | `mpicc -ldress $(pkg-config --libs igraph) -lm` | [`examples/c/mpi_igraph.c`](examples/c/mpi_igraph.c) |
| igraph MPI+CUDA | `dress/mpi/cuda/igraph/dress.h` | `mpicc -ldress -ldress_cuda -lcudart $(pkg-config --libs igraph) -lm` | [`examples/c/mpi_cuda_igraph.c`](examples/c/mpi_cuda_igraph.c) |

```c
// Δ⁰ : edge fingerprint
p_dress_graph_t g = init_dress_graph(N, E, U, V, NULL, DRESS_VARIANT_UNDIRECTED, 0);
int iters; double delta;
dress_fit(g, 100, 1e-6, &iters, &delta);        // CPU
dress_fit_cuda(g, 100, 1e-6, &iters, &delta);   // CUDA

// Δ¹ : histogram fingerprint
int hs; int64_t ns;
int64_t *hist = delta_dress_fit(g, /*k=*/1, 100, 1e-6, &hs, 0, NULL, &ns);
int64_t *hist = delta_dress_fit_mpi(g, 1, 100, 1e-6, &hs, 0, NULL, &ns, MPI_COMM_WORLD);
int64_t *hist = delta_dress_fit_mpi_cuda(g, 1, 100, 1e-6, &hs, 0, NULL, &ns, MPI_COMM_WORLD);

free(hist);
free_dress_graph(g);

// igraph wrapper : same API names, transparent backend switching
#include <dress/igraph/dress.h>           // CPU
#include <dress/cuda/igraph/dress.h>      // CUDA
#include <dress/mpi/igraph/dress.h>       // MPI
#include <dress/mpi/cuda/igraph/dress.h>  // MPI + CUDA

dress_result_igraph_t r;
dress_fit(&graph, NULL, DRESS_VARIANT_UNDIRECTED,
          100, 1e-6, 1, &r);             // same call for all backends
dress_free(&r);

delta_dress_result_igraph_t dr;
delta_dress_fit(&graph, NULL, DRESS_VARIANT_UNDIRECTED,
                /*k=*/1, 100, 1e-6, 1, &dr);  // CPU / MPI / CUDA, per header
delta_dress_free(&dr);
```

### C++

| Backend | Header | Namespace | Example |
|---------|--------|-----------|---------|
| CPU | `dress/dress.hpp` | `::DRESS` | [`examples/cpp/cpu.cpp`](examples/cpp/cpu.cpp) |
| CUDA | `dress/cuda/dress.hpp` | `cuda::DRESS` | [`examples/cpp/cuda.cpp`](examples/cpp/cuda.cpp) |
| MPI | `dress/mpi/dress.hpp` | `mpi::DRESS` | [`examples/cpp/mpi.cpp`](examples/cpp/mpi.cpp) |
| MPI+CUDA | `dress/mpi/cuda/dress.hpp` | `mpi::cuda::DRESS` | [`examples/cpp/mpi_cuda.cpp`](examples/cpp/mpi_cuda.cpp) |

```cpp
DRESS g(N, sources, targets);              // or cuda::DRESS, mpi::DRESS, mpi::cuda::DRESS
auto r = g.fit(100, 1e-6);                 // → {iterations, delta}
auto d = g.deltaFit(/*k=*/1);             // → {histogram, hist_size, num_subgraphs}
double val = g.edgeDress(e);               // per-edge value after fit
```

### Python

| Backend | Import | Example |
|---------|--------|---------|
| CPU | `from dress import dress_fit, delta_dress_fit` | [`cpu.py`](examples/python/cpu.py) |
| CUDA | `from dress.cuda import dress_fit, delta_dress_fit` | [`cuda.py`](examples/python/cuda.py) |
| MPI | `from dress.mpi import delta_dress_fit` | [`mpi.py`](examples/python/mpi.py) |
| MPI+CUDA | `from dress.mpi.cuda import delta_dress_fit` | [`mpi_cuda.py`](examples/python/mpi_cuda.py) |
| NetworkX (CPU) | `from dress.networkx import dress_graph, delta_dress_graph` | [`cpu_nx.py`](examples/python/cpu_nx.py) |
| NetworkX (CUDA) | `from dress.cuda.networkx import dress_graph, delta_dress_graph` | [`cuda_nx.py`](examples/python/cuda_nx.py) |
| NetworkX (MPI) | `from dress.mpi.networkx import delta_dress_graph` | [`mpi_nx.py`](examples/python/mpi_nx.py) |
| NetworkX (MPI+CUDA) | `from dress.mpi.cuda.networkx import delta_dress_graph` | [`mpi_cuda_nx.py`](examples/python/mpi_cuda_nx.py) |
| CPU (OO) | `from dress import DRESS` | [`cpu_oo.py`](examples/python/cpu_oo.py) |
| CUDA (OO) | `from dress.cuda import DRESS` | [`cuda_oo.py`](examples/python/cuda_oo.py) |
| MPI (OO) | `from dress.mpi import DRESS` | [`mpi_oo.py`](examples/python/mpi_oo.py) |
| MPI+CUDA (OO) | `from dress.mpi.cuda import DRESS` | [`mpi_cuda_oo.py`](examples/python/mpi_cuda_oo.py) |

```python
# Δ⁰ : edge fingerprint
result = dress_fit(n_vertices, sources, targets)
result.edge_dress    # per-edge values
result.node_dress    # per-node norms
result.iterations    # convergence iterations

# Δ¹ : histogram fingerprint
result = delta_dress_fit(n_vertices, sources, targets, k=1)
result.histogram     # bin counts
result.hist_size     # number of bins

# NetworkX: pass a graph directly
import networkx as nx
from dress.networkx import dress_graph, delta_dress_graph

G = nx.karate_club_graph()
result = dress_graph(G, set_attributes=True)
G.edges[0, 1]["dress"]      # per-edge similarity

delta = delta_dress_graph(G, k=1, keep_multisets=True)
delta.histogram              # bin counts

# MPI NetworkX: same API, distributed
from dress.mpi.networkx import delta_dress_graph
delta = delta_dress_graph(G, k=1)  # uses MPI.COMM_WORLD
```

### Rust

| Backend | Import | Example |
|---------|--------|---------|
| CPU | `use dress_graph::DRESS` | [`examples/rust/cpu.rs`](examples/rust/cpu.rs) |
| CUDA | `use dress_graph::cuda::DRESS` | [`examples/rust/cuda.rs`](examples/rust/cuda.rs) |
| MPI | `use dress_graph::mpi` | [`examples/rust/mpi.rs`](examples/rust/mpi.rs) |
| MPI+CUDA | `use dress_graph::mpi::cuda` | [`examples/rust/mpi_cuda.rs`](examples/rust/mpi_cuda.rs) |
| CPU (OO) | `use dress_graph::{DRESS, Variant}` | [`examples/rust/cpu_oo.rs`](examples/rust/cpu_oo.rs) |
| CUDA (OO) | `use dress_graph::{cuda, Variant}` | [`examples/rust/cuda_oo.rs`](examples/rust/cuda_oo.rs) |
| MPI (OO) | `use dress_graph::{mpi, Variant}` | [`examples/rust/mpi_oo.rs`](examples/rust/mpi_oo.rs) |
| MPI+CUDA (OO) | `use dress_graph::{mpi, Variant}` | [`examples/rust/mpi_cuda_oo.rs`](examples/rust/mpi_cuda_oo.rs) |

```rust
// Builder pattern (CPU / CUDA)
let result = DRESS::builder(n, sources, targets)
    .variant(Variant::Undirected)
    .build_and_fit()?;
// result.edge_dress, result.iterations, result.delta

// MPI delta fit
let r = mpi::delta_fit(n, sources, targets, None,
    /*k=*/1, 100, 1e-6, Variant::Undirected, false, false, &world)?;
// r.histogram, r.hist_size, r.num_subgraphs
```

### Go

| Backend | Import path | Example |
|---------|-------------|---------|
| CPU | `github.com/velicast/dress-graph/go` | [`examples/go/cpu.go`](examples/go/cpu.go) |
| CUDA | `github.com/velicast/dress-graph/go/cuda` | [`examples/go/cuda.go`](examples/go/cuda.go) |
| MPI | `github.com/velicast/dress-graph/go/mpi` | [`examples/go/mpi.go`](examples/go/mpi.go) |
| MPI+CUDA | `github.com/velicast/dress-graph/go/mpi/cuda` | [`examples/go/mpi_cuda.go`](examples/go/mpi_cuda.go) |
| CPU (OO) | `github.com/velicast/dress-graph/go` | [`examples/go/cpu_oo.go`](examples/go/cpu_oo.go) |
| CUDA (OO) | `github.com/velicast/dress-graph/go/cuda` | [`examples/go/cuda_oo.go`](examples/go/cuda_oo.go) |
| MPI (OO) | `github.com/velicast/dress-graph/go/mpi` | [`examples/go/mpi_oo.go`](examples/go/mpi_oo.go) |
| MPI+CUDA (OO) | `github.com/velicast/dress-graph/go/mpi/cuda` | [`examples/go/mpi_cuda_oo.go`](examples/go/mpi_cuda_oo.go) |

```go
// CPU / CUDA: same function, different import
result, _ := dress.DressFit(n, sources, targets, nil,
    dress.Undirected, 100, 1e-6, false)
// result.EdgeDress, result.Iterations, result.Delta

// MPI / MPI+CUDA: delta only
dress.Init()
defer dress.Finalize()
r, _ := dress.DeltaDressFit(n, sources, targets, nil,
    /*k=*/1, dress.Undirected, 100, 1e-6, false, false)
// r.Histogram, r.HistSize, r.NumSubgraphs
```

### Julia

| Backend | Import | Example |
|---------|--------|---------|
| CPU | `using DRESS` | [`examples/julia/cpu.jl`](examples/julia/cpu.jl) |
| CUDA | `using DRESS.CUDA` | [`examples/julia/cuda.jl`](examples/julia/cuda.jl) |
| MPI | `using DRESS.MPI` | [`examples/julia/mpi.jl`](examples/julia/mpi.jl) |
| MPI+CUDA | `using DRESS.MPI.CUDA` | [`examples/julia/mpi_cuda.jl`](examples/julia/mpi_cuda.jl) |
| CPU (OO) | `using DRESS` | [`examples/julia/cpu_oo.jl`](examples/julia/cpu_oo.jl) |
| CUDA (OO) | `using DRESS.CUDA` | [`examples/julia/cuda_oo.jl`](examples/julia/cuda_oo.jl) |
| MPI (OO) | `using DRESS.MPI` | [`examples/julia/mpi_oo.jl`](examples/julia/mpi_oo.jl) |
| MPI+CUDA (OO) | `using DRESS.MPI.CUDA` | [`examples/julia/mpi_cuda_oo.jl`](examples/julia/mpi_cuda_oo.jl) |

```julia
# Δ⁰ : edge fingerprint
r = dress_fit(N, sources, targets)
# r.edge_dress, r.node_dress, r.iterations, r.delta

# Δ¹ : histogram fingerprint
r = delta_dress_fit(N, sources, targets; k=1)
# r.histogram, r.hist_size, r.num_subgraphs
```

### R

| Backend | Call | Example |
|---------|------|---------|
| CPU | `dress_fit()` / `delta_dress_fit()` | [`examples/r/cpu.R`](examples/r/cpu.R) |
| CUDA | `cuda$dress_fit()` / `cuda$delta_dress_fit()` | [`examples/r/cuda.R`](examples/r/cuda.R) |
| MPI | `mpi$delta_dress_fit()` | [`examples/r/mpi.R`](examples/r/mpi.R) |
| MPI+CUDA | `mpi$cuda$delta_dress_fit()` | [`examples/r/mpi_cuda.R`](examples/r/mpi_cuda.R) |
| CPU (OO) | `DRESS(...)$fit()` | [`examples/r/cpu_oo.R`](examples/r/cpu_oo.R) |
| CUDA (OO) | `cuda$DRESS(...)$fit()` | [`examples/r/cuda_oo.R`](examples/r/cuda_oo.R) |
| MPI (OO) | `mpi$DRESS(...)$delta_fit()` | [`examples/r/mpi_oo.R`](examples/r/mpi_oo.R) |
| MPI+CUDA (OO) | `mpi$cuda$DRESS(...)$delta_fit()` | [`examples/r/mpi_cuda_oo.R`](examples/r/mpi_cuda_oo.R) |

```r
library(dress.graph)

# Δ⁰ : edge fingerprint
r <- dress_fit(6L, sources, targets)
# r$edge_dress, r$node_dress, r$iterations, r$delta

# Δ¹ : histogram fingerprint
r <- delta_dress_fit(6L, sources, targets, k = 1L)       # CPU
r <- cuda$delta_dress_fit(6L, sources, targets, k = 1L)   # CUDA
r <- mpi$delta_dress_fit(6L, sources, targets, k = 1L)    # MPI
r <- mpi$cuda$delta_dress_fit(6L, sources, targets, k = 1L) # MPI+CUDA
```

### MATLAB / Octave

| Backend | Function | Example |
|---------|----------|---------|
| CPU | `dress_fit()` / `delta_dress_fit()` | [`examples/octave/cpu.m`](examples/octave/cpu.m) |
| CUDA | `cuda.dress_fit()` | [`examples/octave/cuda_example.m`](examples/octave/cuda_example.m) |
| Δ¹-DRESS | `delta_dress_fit(..., 'K', 1)` | [`examples/octave/rook_vs_shrikhande.m`](examples/octave/rook_vs_shrikhande.m) |
| CPU (OO) | `DRESS(...)` | [`examples/octave/cpu_oo.m`](examples/octave/cpu_oo.m) |
| CPU (OO, Matlab) | `DRESS(...)` | [`examples/matlab/cpu_oo.m`](examples/matlab/cpu_oo.m) |
| CUDA (OO, Matlab) | `cuda.DRESS(...)` | [`examples/matlab/cuda_oo.m`](examples/matlab/cuda_oo.m) |
| MPI (OO, Matlab) | `mpi.DRESS(...)` | [`examples/matlab/mpi_oo.m`](examples/matlab/mpi_oo.m) |
| MPI+CUDA (OO, Matlab) | `mpi.cuda.DRESS(...)` | [`examples/matlab/mpi_cuda_oo.m`](examples/matlab/mpi_cuda_oo.m) |

```matlab
% Δ⁰ : edge fingerprint
result = dress_fit(6, int32(sources), int32(targets));
% result.edge_dress, result.node_dress, result.iterations, result.delta

% Δ¹ : histogram fingerprint
result = delta_dress_fit(6, int32(sources), int32(targets), ...
    'K', 1, 'KeepMultisets', true);
% result.histogram, result.hist_size, result.multisets, result.num_subgraphs

% Persistent graph with get()
g = DRESS(6, int32(sources), int32(targets));
g.fit('MaxIterations', 100, 'Epsilon', 1e-6);
d = g.get(u, v);     % query edge similarity
g.close();
```

### JavaScript / WASM

| Backend | Import | Example |
|---------|--------|---------|
| CPU | `import { dressFit } from './dress.js'` | [`examples/wasm/cpu.mjs`](examples/wasm/cpu.mjs) |
| Δ¹-DRESS | `import { deltaDressFit } from './dress.js'` | [`examples/wasm/rook_vs_shrikhande.mjs`](examples/wasm/rook_vs_shrikhande.mjs) |
| CPU (OO) | `import { DressGraph } from 'dress-graph'` | [`examples/wasm/cpu_oo.mjs`](examples/wasm/cpu_oo.mjs) |

```javascript
// Δ⁰ : edge fingerprint
const result = await dressFit({
    numVertices: 6, sources, targets,
});
// result.edgeDress, result.nodeDress, result.iterations, result.delta

// Δ¹ : histogram fingerprint
const r = await deltaDressFit({
    numVertices: 6, sources, targets,
    k: 1, keepMultisets: true,
});
// r.histogram, r.histSize, r.multisets, r.numSubgraphs

// Persistent graph with get()
const g = await DressGraph.create({ numVertices: 6, sources, targets });
g.fit(100, 1e-6);
const d = g.get(u, v);   // query edge similarity
g.free();
```

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

The arXiv papers is outdated and will be updated next week.

- E. Castrillo. *DRESS: A Continuous Framework for Structural Graph Refinement.* [arXiv:2602.20833](https://github.com/velicast/dress-graph/blob/main/research/k-DRESS.pdf)
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

---

> **Why "DRESS"?** DRESS computes an edge labeling that reveals the graph's hidden
> structural identity - it *dresses* the bare skeleton (adjacency) with
> meaningful values.  A graph without DRESS is "naked" topology; after DRESS,
> every edge wears the structural role that fits it best.  And `dress_fit()`
> is literally fitting the dress to the graph: few iterations give a loose fit,
> more iterations tighten it, and at steady state the fit is true to size.

## License

[MIT](LICENSE)
