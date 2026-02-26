# dress-graph

**A Continuous Framework for Structural Graph Refinement**

DRESS is a parameter-free algorithm that computes a unique, self-consistent
edge similarity for any graph.  Given an edge list, it iteratively solves a
nonlinear fixed-point system where every edge's value depends on its
neighbours' values.  The result is bounded in [0, 2], deterministic, and
requires no tuning.  Sorting the edge values produces a canonical **graph fingerprint**.

For the theory, generalizations (Motif-DRESS, Δ-DRESS), see the research paper:
[**arXiv:2602.20833**](https://arxiv.org/abs/2602.20833)

For the relationship between DRESS and the Weisfeiler-Leman hierarchy — each
deletion level ℓ empirically matches (ℓ+2)-WL on CFI graphs:
[**arxiv:2602.21557**](https://arxiv.org/abs/2602.21557)

## Key properties

| Property |
|----------|
| Bounded [0, 2], self-similarity = 2 |
| Parameter-free (no damping factor) |
| Scale invariant (degree-0 homogeneous) |
| Unique deterministic fixed point |
| Low complexity: O(E) per iteration, O(N + E) memory |
| Massively parallelisable (per-edge independent updates) |

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

## Current Applications

- **Graph Isomorphism**: sorting DRESS edge values produces a canonical fingerprint. 100 % accuracy on MiVIA and IsoBench benchmarks.
- **Community Detection**: DRESS values classify edges as intra- or inter-community, improving SCAN and enabling agglomerative hierarchical clustering.
- **Classification**: percentile-based DRESS fingerprints fed to standard classifiers match or exceed Weisfeiler-Leman baselines on TU benchmark datasets.
- **Retrieval**: DRESS fingerprint distances correlate strongly with graph edit distance, achieving state-of-the-art precision on GED-based retrieval benchmarks.
- **GED Regression**: DRESS fingerprint differences fed to a simple regressor predict graph edit distance with 15× lower MSE than TaGSim on LINUX graphs — no GNN required.
- **Edge Robustness**: DRESS edge values double as an O(km) edge-importance ranking that outperforms O(nm) betweenness centrality and four other baselines (65–97% win rates, p < 0.0001 across 224 graphs).

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

### Δ^k-DRESS (deletion-based refinement)

```python
from dress import delta_dress_fit

result = delta_dress_fit(
    n_vertices=4,
    sources=[0, 1, 2, 0],
    targets=[1, 2, 3, 3],
    k=1,              # remove 1 vertex at a time
    epsilon=1e-6,
)
print(result.histogram)   # histogram of edge DRESS values across all subgraphs
print(result.hist_size)   # number of bins
```

## Language bindings

DRESS is implemented in C with bindings for:

- **C / C++**: static and shared libraries
- **Python**: pybind11 (`pip install dress-graph`)
- **Rust**: `dress-graph` crate
- **Go**: CGo bindings
- **Julia**: native FFI module
- **R**: `.Call` interface
- **MATLAB / Octave**: MEX gateway
- **JavaScript / WASM**: browser and Node.js

## Building from source

```bash
# C / C++ (CMake)
mkdir build && cd build
cmake .. && make

# Rust
cd rust && cargo build --release

# Python
cd python && pip install .
```

## Documentation

Full documentation (theory, applications, API reference):
[https://velicast.github.io/dress-graph/](https://velicast.github.io/dress-graph/)

## Publications

- E. Castrillo. *DRESS and the WL Hierarchy: Climbing One Deletion at a Time.* [research/delta-k-DRESS.pdf](https://arxiv.org/abs/2602.21557)
- E. Castrillo. *DRESS: A Continuous Framework for Structural Graph Refinement.* [arXiv:2602.20833](https://arxiv.org/abs/2602.20833)
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
```

---

> **Why "DRESS"?** DRESS computes an edge labelling that reveals the graph's hidden
> structural identity — it *dresses* the bare skeleton (adjacency) with
> meaningful values.  A graph without DRESS is "naked" topology; after DRESS,
> every edge wears the structural role that fits it best.  And `dress_fit()`
> is literally fitting the dress to the graph: few iterations give a loose fit,
> more iterations tighten it, and at steady state the fit is true to size.

## License

[MIT](LICENSE)
