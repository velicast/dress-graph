# dress-graph

**Diffusive Recursive Structural Similarity on Graphs**

DRESS is a parameter-free algorithm that computes a unique, self-consistent
edge similarity for any graph.  Given an edge list, it iteratively solves a
nonlinear fixed-point system where every edge's value depends on its
neighbours' values.  The result is bounded in [0, 2], deterministic, and
requires no tuning.  Sorting the edge values produces a **graph fingerprint**
that achieves 100 % accuracy on standard isomorphism benchmarks.

## Key properties

| Property | Status |
|----------|--------|
| Bounded [0, 2], self-similarity = 2 | Proven |
| Parameter-free (no damping factor) | Proven |
| Scale invariant (degree-0 homogeneous) | Proven |
| Unique deterministic fixed point | Proven |
| Low complexity: O(E) per iteration, O(N + E) memory | Proven |
| Massively parallelisable (per-edge independent updates) | Implemented |

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

> **Why "DRESS"?** The acronym (**D**iffusive **R**ecursive
> **E**dge **S**tructural **S**imilarity) is also a nod to what it
> does: DRESS computes an edge labelling that reveals the graph's hidden
> structural identity — it *dresses* the bare skeleton (adjacency) with
> meaningful values.  A graph without DRESS is "naked" topology; after DRESS,
> every edge wears the structural role that fits it best.  And `dress_fit()`
> is literally fitting the dress to the graph: few iterations give a loose fit,
> more iterations tighten it, and at steady state the fit is true to size.

## License

[MIT](LICENSE)
