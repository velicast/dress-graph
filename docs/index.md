# DRESS Graph

**A Continuous Framework for Structural Graph Refinement**

DRESS is a provably (Under reconstruction conjecture) continuous relaxation of the Weisfeiler–Leman algorithm.
At depth \(k\), higher-order DRESS is **provably at least as powerful as \((k{+}2)\)-WL**
in expressiveness - the base algorithm (\(k{=}0\)) already matches 2-WL, and each
level adds one WL dimension.
Yet it is dramatically cheaper to compute: a single DRESS run costs
\(\mathcal{O}(I \cdot m \cdot d_{\max})\) where \(I\) is the number of iterations,
and depth-\(k\) requires \(\binom{n}{k}\) independent runs - a total of
\(\mathcal{O}\bigl(\binom{n}{k} \cdot I \cdot m \cdot d_{\max}\bigr)\),
compared to \(\mathcal{O}(n^{k+3})\) for \((k{+}2)\)-WL.
Space complexity is \(\mathcal{O}(n + m)\), compared to \(\mathcal{O}(n^{k+2})\) for \((k{+}2)\)-WL.
The algorithm is embarrassingly parallel in two orthogonal ways -
across the \(\binom{n}{k}\) subproblems and across edge updates within each iteration -
enabling distributed/cloud plus multi-core/GPU/SIMD implementations.

DRESS is a parameter-free algorithm that computes a unique, self-consistent
edge similarity for any graph.  Given an edge list, it iteratively solves a
nonlinear fixed-point system where every edge's similarity value depends on
its neighbors' values.  It converges to a unique, bounded solution in
\([0, 2]\) for unweighted graphs (values may exceed 2 with non-uniform edge
weights) with no tuning parameters.  Sorting the edge values produces a
canonical **graph fingerprint**.

For the theory and generalizations (DRESS Family), see the research paper:
[**arXiv:2602.20833**](https://github.com/velicast/dress-graph/blob/main/research/k-DRESS.pdf)

For the relationship between DRESS and the Weisfeiler–Leman hierarchy:
[**arXiv:2602.21557**](https://github.com/velicast/dress-graph/blob/main/research/vertex-k-DRESS.pdf)

---

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
| Edge-centric refinement (operates only on edges) |
| Parameter-free core (no damping factor, no hyperparameters) |
| Unique fixed point via Birkhoff contraction |
| Bounded exactly in \([0, 2]\) for unweighted graphs, self-similarity \(d_{uu} = 2\) |
| Isomorphism-invariant |
| Scale-invariant (degree-0 homogeneous) |
| Completely deterministic |
| Practical convergence in ≤ 20 iterations |
| Continuous, ML-usable fingerprints (sorted values / ε-binned histogram) |
| Theoretical per-iteration \(\mathcal{O}(\|E\|)\), memory \(\mathcal{O}(\|V\| + \|E\|)\) |
| Massively parallelizable (\(\Delta^k\) subproblems and per-edge updates) |
| Native weighted-graph support via symmetric weight function |
| Supports directed graphs (four variants: undirected, directed, forward, backward) |
| Provably numerically stable (no overflows, no undefined behaviors) |
| Self-loops on every node (closed neighborhoods ensure \(\|u\| > 0\) and well-defined iteration on all graphs, including sparse/isolated edges) |

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

## Current Experimented Applications

- **Graph Isomorphism**: sorting DRESS edge values produces a canonical fingerprint. 100 % accuracy on MiVIA and IsoBench benchmarks.
- **Community Detection**: DRESS values classify edges as intra- or inter-community, improving SCAN and enabling agglomerative hierarchical clustering.
- **Classification**: percentile-based DRESS fingerprints fed to standard classifiers match or exceed Weisfeiler-Leman baselines on TU benchmark datasets.
- **Retrieval**: DRESS fingerprint distances correlate strongly with graph edit distance, achieving state-of-the-art precision on GED-based retrieval benchmarks.
- **GED Regression**: DRESS fingerprint differences fed to a simple regressor predict graph edit distance with 15× lower MSE than TaGSim on LINUX graphs - no GNN required.
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
    edge labeling that reveals the graph's hidden structural identity. It
    *dresses* the bare skeleton (adjacency) with meaningful values.  A graph
    without DRESS is "naked" topology; after DRESS, every edge wears the
    structural role that fits it best.  And `dress_fit()` is literally fitting
    the dress to the graph: few iterations give a loose fit, more iterations
    tighten it, and at steady state the fit is true to size.
