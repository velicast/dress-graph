# DRESS Graph

**A Continuous Framework for Structural Graph Refinement**

DRESS is a parameter-free, self-regularizing algorithm that computes a unique
self-consistent edge similarity for any graph.  Given an edge list, DRESS
iteratively solves a nonlinear fixed-point system where every edge's similarity
value depends on its neighbors' values.  It converges to a unique, bounded
solution in \([0, 2]\) with no tuning parameters.

This work is an independent continuation of research from the author's master's thesis:

> **A fast heuristic algorithm for community detection in large-scale complex networks**
> Eduar Castrillo, Universidad Nacional de Colombia, 2018
> [bdigital.unal.edu.co/69933](http://bdigital.unal.edu.co/69933/)

DRESS emerged from intuition while developing algorithms for community
detection.  What started as an edge scoring function turned out to be a
fundamental structural object: a single algorithm that simultaneously enables
graph isomorphism testing, community detection, classification, retrieval,
and edge-importance ranking.

!!! note "On the name"
    DRESS was originally named **DSS** (Dynamic Structural
    Similarity), then renamed to **DRESS**
    to emphasize the diffusive and recursive
    nature of the equation.  The word *Dynamic* in the original name referred
    to the fact that the underlying fixed-point system is a discrete dynamical
    system, but it was dropped to avoid confusion with *dynamic graphs*
    (graphs that change over time), which are a different concept entirely.

---

## Key properties

| Property |
|----------|
| Bounded \([0, 2]\), self-similarity \(= 2\) |
| Parameter-free (self-regularizing, no damping factor) |
| Scale invariant (degree-0 homogeneous) |
| Unique deterministic fixed point |
| Numerically stable (no overflow, no error amplification) |
| Low complexity: \(O(E)\) per iteration, \(O(N + E)\) memory |
| Graph fingerprinting (practical): MiVIA / IsoBench 100 % |
| Community detection (improves SCAN) |
| Weighted + directed support |
| Massively parallelizable (per-edge independent updates) |

## The equation at a glance

For each edge \((u, v)\) in the graph, the **fixed-point equation** is:

\[
d_{uv} = \frac{\displaystyle\sum_{x \in N[u] \cap N[v]}
  \bigl(\bar{w}_{ux}\, d_{ux} + \bar{w}_{vx}\, d_{vx}\bigr)}
  {\|u\| \cdot \|v\|}
\]

The corresponding **discrete-time iteration** is:

\[
d_{uv}^{(t+1)} = \frac{\displaystyle\sum_{x \in N[u] \cap N[v]}
  \bigl(\bar{w}_{ux}\, d_{ux}^{(t)} + \bar{w}_{vx}\, d_{vx}^{(t)}\bigr)}
  {\|u\|^{(t)} \cdot \|v\|^{(t)}}
\]

where \(\|u\|^{(t)} = \sqrt{\displaystyle\sum_{x \in N[u]} \bar{w}_{ux}\, d_{ux}^{(t)}}\),
\(N[u] = N(u) \cup \{u\}\) is the closed neighborhood (including a
self-loop with \(\bar{w}_{uu} = 2\), \(d_{uu} = 2\)), and \(\bar{w}\) is the
**combined weight** for the chosen variant.

See [The DRESS Equation](theory/equation.md) for the full derivation.

## Current Applications

- **Graph Isomorphism**: sorting DRESS edge values produces a canonical fingerprint. 100 % accuracy on MiVIA and IsoBench benchmarks.
- **Community Detection**: DRESS values classify edges as intra- or inter-community, improving SCAN and enabling agglomerative hierarchical clustering.
- **Classification**: percentile-based DRESS fingerprints fed to standard classifiers match or exceed Weisfeiler-Leman baselines on TU benchmark datasets.
- **Retrieval**: DRESS fingerprint distances correlate strongly with graph edit distance, achieving state-of-the-art precision on GED-based retrieval benchmarks.
- **GED Regression**: DRESS fingerprint differences fed to a simple regressor predict graph edit distance with 15× lower MSE than TaGSim on LINUX graphs — no GNN required.
- **Edge Robustness**: DRESS edge values double as an O(km) edge-importance ranking that outperforms O(nm) betweenness centrality and four other baselines (65–97% win rates, p < 0.0001 across 224 graphs).

See [Applications Overview](applications/overview.md) for details.

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

See [Installation](getting-started/installation.md) to get started.

---

!!! quote "Why 'DRESS'?"
    DRESS computes an
    edge labelling that reveals the graph's hidden structural identity. It
    *dresses* the bare skeleton (adjacency) with meaningful values.  A graph
    without DRESS is "naked" topology; after DRESS, every edge wears the
    structural role that fits it best.  And `dress_fit()` is literally fitting
    the dress to the graph: few iterations give a loose fit, more iterations
    tighten it, and at steady state the fit is true to size.
