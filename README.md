# dress-graph

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

PyPi [![PyPI Downloads](https://static.pepy.tech/personalized-badge/dress-graph?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/dress-graph) Crates [![crates.io downloads](https://img.shields.io/crates/d/dress-graph)](https://crates.io/crates/dress-graph) NPM [![npm downloads](https://img.shields.io/npm/dt/dress-graph)](https://www.npmjs.com/package/dress-graph) CRAN [![CRAN downloads](https://cranlogs.r-pkg.org/badges/grand-total/dress.graph)](https://cran.r-project.org/package=dress.graph)

Quickstart [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/velicast/dress-graph/blob/main/notebooks/quickstart.ipynb)
Prism vs K₃,₃ [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/velicast/dress-graph/blob/main/notebooks/prism_vs_k33.ipynb)
Rook vs Shrikhande [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/velicast/dress-graph/blob/main/notebooks/delta_dress_rook_shrikhande.ipynb)

## A Continuous Framework for Structural Graph Refinement

We introduce DRESS, a deterministic, parameter-free framework that iteratively refines the structural similarity of edges in a graph to produce a *canonical fingerprint*: a real-valued edge vector, obtained by converging a non-linear dynamical system to its unique fixed point. The fingerprint is *isomorphism-invariant* by construction, *numerically stable* (all values lie in [0, 2]), *fast* and *embarrassingly parallel* to compute: each iteration costs O(m · d_max) and convergence is guaranteed by Birkhoff contraction. As a direct consequence of these properties, DRESS is provably at least as expressive as the 2-dimensional Weisfeiler–Leman (2-WL) test, at a fraction of the cost (O(m · d_max) vs. O(n³) per iteration). We generalize the original equation (Castrillo, León, and Gómez, 2018) to Motif-DRESS (arbitrary structural motifs) and Generalized-DRESS (abstract aggregation template), and introduce Δ-DRESS, which runs DRESS on each vertex-deleted subgraph to boost expressiveness. Δ-DRESS empirically separates all 7,983 graphs in a comprehensive Strongly Regular Graph benchmark, and iterated deletion (Δᵏ-DRESS) climbs the CFI staircase, achieving (k+2)-WL expressiveness at each depth k. Successfully applied to anoter handful of downstream applications. The algorithm is embarrassingly parallel in two orthogonal ways - across the vertex-deleted subgraphs and across edge updates within each iteration - enabling distributed/cloud plus multi-core/GPU/SIMD implementations.

> **Note on Wrappers:** Please report any bugs you find while using the language wrappers (Python, Rust, JS, etc.). I am moving quickly and relying on AI to speed up the development of the wrappers, but I am directly and carefully maintaining the core C backend.

The arXiv papers are outdated and will be updated next week. The latest versions including the proof in Paper 2, are in the GitHub repo.

For the theory and generalizations (DRESS Family), see the research paper:
[**arXiv:2602.20833**](https://github.com/velicast/dress-graph/blob/main/research/k-DRESS.pdf)

For the relationship between DRESS and the Weisfeiler–Leman hierarchy:
[**arXiv:2602.21557**](https://github.com/velicast/dress-graph/blob/main/research/vertex-k-DRESS.pdf)

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
| Practical convergence by few iterations (contraction)|
| Continuous canonical fingerprints (sorted values / ε-binned histogram) |
| Theoretical per-iteration $\mathcal{O}(\|E\|)$, memory $\mathcal{O}(\|V\| + \|E\|)$ |
| Massively parallelizable ($\Delta^k$ subproblems and per-edge updates) |
| Native weighted-graph support via symmetric weight function |
| Supports directed graphs (four variants: undirected, directed, forward, backward) |
| Provably numerically stable (no overflows, no undefined behaviors) |

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

  **Strongly Regular Graphs — Δ¹-DRESS (7,983 graphs, 100 % separated)**

  Plain DRESS (Δ⁰) assigns a single uniform value to every edge in an SRG, producing zero separation. Δ¹-DRESS breaks this symmetry by running DRESS on each vertex-deleted subgraph:

  | Family | Parameters | Graphs | Δ¹ unique | Separated | Min L∞ |
  |--------|-----------|:------:|:---------:|:---------:|:------:|
  | Conference (Mathon) | (45, 22, 10, 11) | 6 | 6 | **100 %** | 4.16 × 10⁻³ |
  | Steiner block S(2,4,28) | (63, 32, 16, 16) | 4,466 | 4,466 | **100 %** | 1.95 × 10⁻³ |
  | Quasi-symmetric 2-designs | (63, 32, 16, 16) | 3,511 | 3,511 | **100 %** | 2.23 × 10⁻³ |

  SRG data from [Krystal Guo's repository](https://github.com/kguo-sagecode/Strongly-regular-graphs). Min L∞ is the closest-pair distance (1,000 random pairs); separation is stable across all rounding precisions 6d–14d.

  **CFI Staircase — Δᵏ-DRESS climbs the WL hierarchy**

  The [CFI construction](https://en.wikipedia.org/wiki/Cai%E2%80%93F%C3%BCrer%E2%80%93Immerman_graph) produces the canonical hard instances for every WL level. Δᵏ-DRESS matches $(k{+}2)$-WL on each:

  | Base graph | \|V(CFI)\| | WL req. | Δ⁰ | Δ¹ | Δ² | Δ³ |
  |:----------:|:----------:|:-------:|:--:|:--:|:--:|:--:|
  | $K_3$ | 6 | 2-WL | ✓ | ✓ | ✓ | ✓ |
  | $K_4$ | 16 | 3-WL | ✗ | ✓ | ✓ | ✓ |
  | $K_5$ | 40 | 4-WL | ✗ | ✗ | ✓ | ✓ |
  | $K_6$ | 96 | 5-WL | ✗ | ✗ | ✗ | ✓ |
  | $K_7$ | 224 | 6-WL | ✗ | ✗ | ✗ | ✗ |

  Each deletion level adds exactly one WL dimension. See [Paper 2](https://github.com/velicast/dress-graph/blob/main/research/vertex-k-DRESS.pdf) for proofs and the full table up to $K_{10}$.

  **Standard benchmarks — Original-DRESS (Δ⁰)**

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
- **JavaScript / WASM**: browser and Node.js

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

The arXiv papers are outdated and will be updated next week. The latest versions including the proof in Paper 2, are in the GitHub repo.

- E. Castrillo. *DRESS and the WL Hierarchy: Climbing One Deletion at a Time.* [arxiv:2602.21557](https://github.com/velicast/dress-graph/blob/main/research/vertex-k-DRESS.pdf)
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

@misc{castrillo2026kdress,
  title   = {DRESS and the WL Hierarchy: Climbing One Deletion at a Time},
  author  = {Eduar Castrillo Velilla},
  year    = {2026},
  eprint  = {2602.21557},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DS},
  url     = {https://arxiv.org/abs/2602.21557}
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
