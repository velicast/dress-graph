# DRESS Graph

**Diffusive Recursive Structural Similarity on Graphs**

DRESS is a parameter-free, self-regularising algorithm that computes a unique
self-consistent edge similarity for any graph.  Given an edge list, DRESS
iteratively solves a nonlinear fixed-point system where every edge's similarity
value depends on its neighbours' values.  It converges to a unique, bounded
solution in \([0, 2]\) with no tuning parameters.

This work is a continuation of the research from the master's thesis:

> **A fast heuristic algorithm for community detection in large-scale complex networks**
> Eduar Castrillo, Universidad Nacional de Colombia, 2018
> [bdigital.unal.edu.co/69933](http://bdigital.unal.edu.co/69933/)

DRESS emerged from intuition while developing algorithms for community
detection.  What started as an edge scoring function turned out to be a
fundamental structural object: a single formula that simultaneously enables
graph isomorphism testing, community detection, and graph fingerprinting.

!!! note "On the name"
    DRESS was originally named **DSS** (**D**ynamic **S**tructural
    **S**imilarity), then renamed to **DRESS** (**D**iffusive **R**ecursive
    **S**tructural **S**imilarity) to emphasise the diffusive and recursive
    nature of the equation.  The word *Dynamic* in the original name referred
    to the fact that the underlying fixed-point system is a discrete dynamical
    system, but it was dropped to avoid confusion with *dynamic graphs*
    (graphs that change over time), which are a different concept entirely.

---

## Key properties

| Property | Status |
|----------|--------|
| Bounded \([0, 2]\), self-similarity \(= 2\) | Proven |
| Parameter-free (self-regularising, no damping factor) | Proven |
| Scale invariant (degree-0 homogeneous) | Proven |
| Unique deterministic fixed point | Proven |
| Numerically stable (no overflow, no error amplification) | Proven |
| Low complexity: \(O(E)\) per iteration, \(O(N + E)\) memory | Proven |
| Graph fingerprinting (practical) | MiVIA / IsoBench 100 % |
| Community detection (improves SCAN) | Proven |
| Weighted + directed support | Implemented |
| Massively parallelisable (per-edge independent updates) | Implemented |

## The formula at a glance

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
\(N[u] = N(u) \cup \{u\}\) is the closed neighbourhood (including a
self-loop with \(\bar{w}_{uu} = 2\), \(d_{uu} = 2\)), and \(\bar{w}\) is the
**combined weight** for the chosen variant.

See [The DRESS Equation](theory/equation.md) for the full derivation.

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
    The acronym (**D**iffusive **R**ecursive **E**dge **S**tructural
    **S**imilarity) is also a nod to what it does: DRESS computes an
    edge labelling that reveals the graph's hidden structural identity. It
    *dresses* the bare skeleton (adjacency) with meaningful values.  A graph
    without DRESS is "naked" topology; after DRESS, every edge wears the
    structural role that fits it best.  And `dress_fit()` is literally fitting
    the dress to the graph: few iterations give a loose fit, more iterations
    tighten it, and at steady state the fit is true to size.
