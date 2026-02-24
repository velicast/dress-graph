# dress-graph

**A Continuous Framework for Structural Graph Refinement**

DRESS is a parameter-free algorithm that computes a unique, self-consistent
edge similarity for any graph.  Given an edge list, it iteratively solves a
nonlinear fixed-point system where every edge's value depends on its
neighbours' values.  The result is bounded in [0, 2], deterministic, and
requires no tuning.  Sorting the edge values produces a **graph fingerprint**
that achieves 100 % accuracy on standard isomorphism benchmarks.

For the theory, generalizations (Motif-DRESS, Δ-DRESS), see the research paper:
[**research/k-DRESS.pdf**](research/k-DRESS.pdf)

For iterated deletion (Δℓ-DRESS) and climbing the WL hierarchy on CFI graphs:
[**research/delta-k-DRESS.pdf**](research/delta-k-DRESS.pdf)

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
from dress import DRESS, UNDIRECTED

g = DRESS(
    n_vertices=4,
    sources=[0, 1, 2, 0],
    targets=[1, 2, 3, 3],
    variant=UNDIRECTED,
)
result = g.fit()
print(g.dress_values)  # DRESS value for each edge
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

- E. Castrillo. *Δℓ-DRESS: Climbing the WL Hierarchy One Deletion at a Time.* [research/delta-k-DRESS.pdf](research/delta-k-DRESS.pdf)
- E. Castrillo. *DRESS: A Continuous Framework for Structural Graph Refinement.* [research/k-DRESS.pdf](research/k-DRESS.pdf)
- E. Castrillo, E. León, J. Gómez. *Dynamic Structural Similarity on Graphs.* [arXiv:1805.01419](https://arxiv.org/abs/1805.01419)
- E. Castrillo, E. León, J. Gómez. *Fast Heuristic Algorithm for Multi-Scale Hierarchical Community Detection.* [ASONAM 2017](https://dl.acm.org/citation.cfm?doid=3110025.3110125)
- E. Castrillo, E. León, J. Gómez. *High-Quality Disjoint and Overlapping Community Structure in Large-Scale Complex Networks.* [arXiv:1805.12238](https://arxiv.org/abs/1805.12238)

## Citing

If you use DRESS in your research, please cite:

```bibtex
@misc{castrillo2018dress,
  title   = {Dynamic Structural Similarity on Graphs},
  author  = {Eduar Castrillo and Elizabeth Le\'{o}n and Jonatan G\'{o}mez},
  year    = {2018},
  eprint  = {1805.01419},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SI},
  url     = {https://arxiv.org/abs/1805.01419}
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
