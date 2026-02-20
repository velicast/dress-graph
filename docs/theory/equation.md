# The DRESS Equation

## Motivation

Most graph similarity measures are either closed-form formulae (common
neighbours, Jaccard) or linear eigenvector problems (PageRank, spectral
methods).  DRESS is neither. It is a **self-referential nonlinear fixed point
over edges**.

Every edge's similarity depends on its neighbours' similarities, and the
normalising denominator itself depends on edge values.  Despite this circular
dependency, the system converges to a unique solution.

## Definition

For each edge \((u, v) \in E\), define the **closed neighbourhood**
\(N[u] = N(u) \cup \{u\}\), where the self-loop carries combined weight
\(\bar{w}_{uu} = 2\) and similarity \(d_{uu} = 2\).

\[
d_{uv} = \frac{\displaystyle\sum_{x \in N[u] \cap N[v]}
  \bigl(\bar{w}_{ux}\, d_{ux} + \bar{w}_{vx}\, d_{vx}\bigr)}
  {\|u\| \cdot \|v\|}
\]

where the **node norm** is:

\[
\|u\| = \sqrt{\displaystyle\sum_{x \in N[u]} \bar{w}_{ux}\, d_{ux}}
\]

The numerator sums over every common neighbour \(x\) of \(u\) and \(v\)
(including self-loops), adding the weighted similarities that \(u\) and \(v\)
each have with \(x\).  The denominator normalises by the geometric mean of
the two node norms, ensuring the result is bounded.

## Convergence

The iteration is:

1. Initialise all \(d_{uv}^{(0)} = c\) for any constant \(c \ge 0\)
   (typically \(c = 1\)).
2. For each iteration \(t\), compute \(d_{uv}^{(t+1)}\) from \(d^{(t)}\)
   using the equation above.
3. Stop when \(\max_{(u,v)} |d_{uv}^{(t+1)} - d_{uv}^{(t)}| < \epsilon\).

The fixed point is **unique**: the iteration converges to the same solution
regardless of the initial value \(c\).  Empirically, convergence is reached
in 5–20 iterations.

## Variants

For directed graphs, four adjacency constructions are supported.  Each variant
determines both the neighbourhood and the combined edge weight:

| Variant | Neighbourhood \(N[u]\) | Combined weight \(\bar{w}(u,v)\) |
|---------|------------------------|----------------------------------|
| `UNDIRECTED` | \(\{u\} \cup\) all neighbours (ignoring direction) | \(2\,w(u,v)\) |
| `DIRECTED` | \(\{u\} \cup\) all neighbours (in + out) | \(w(u,v) + w(v,u)\) |
| `FORWARD` | \(\{u\} \cup\) out-neighbours | \(w(u,v)\) |
| `BACKWARD` | \(\{u\} \cup\) in-neighbours | \(w(v,u)\) |

## Complexity

### Time

Let \(N = |V|\), \(E = |E|\), and \(T\) be the total number of
common-neighbour entries across all edges.

| Phase | Without intercepts | With precomputed intercepts |
|-------|-------------------|-----------------------------|
| **Initialisation** (CSR build, self-loops) | \(O(N + E)\) | \(O(N + E + T)\) |
| **Per iteration** | \(O\!\left(\sum_{(u,v)} (\deg u + \deg v)\right)\) | \(O(T)\) |
| **Total** (\(k\) iterations) | \(O(N + k \cdot E \cdot \bar{d})\) | \(O(N + E + k \cdot T)\) |

where \(\bar{d}\) is the average degree.  In sparse graphs
(\(E = O(N)\)), the per-iteration cost is \(O(N \cdot \bar{d})\); in
practice \(k \le 20\), so the total is effectively **linear in the
graph size**.

With precomputed intercepts the per-edge update cost drops from
\(O(\deg u + \deg v)\) to \(O(|N[u] \cap N[v]|)\), which is
substantially faster on sparse graphs where most neighbours are *not*
shared.

### Memory

| Component | Bytes |
|-----------|-------|
| CSR adjacency + edge arrays (no intercepts) | \(12N + 48E\) |
| CSR adjacency + edge arrays + intercepts | \(12N + 52E + 8T\) |

The dominant term is \(O(E)\) without intercepts or \(O(E + T)\) with
them.  For sparse graphs \(T = O(E \cdot \bar{d})\).

## Implementation: edge-level parallelism

Each edge's update in a given iteration reads only the **previous
iteration's** values. There are no read–write dependencies between
edges within the same iteration.  This makes DRESS embarrassingly
parallel at the edge level:

- **CPU (OpenMP).** The reference C implementation parallelises the
  edge loop with `#pragma omp parallel for`.  Each thread processes a
  subset of edges independently.
- **GPU (CUDA / OpenCL / Metal).** Each edge can be mapped to a single
  thread or warp.  The CSR adjacency and intercept arrays are
  read-only during the iteration, so they can reside in device
  memory with no synchronisation.  Only a barrier between iterations
  (to swap the read/write buffers) is needed.
- **Distributed.** Edges can be partitioned across machines.  Each
  partition needs read access to the edge values of its neighbours
  (a halo / ghost layer), exchanged once per iteration.  The
  communication volume scales with the number of *cut edges*
  between partitions, not the total graph size.

The double-buffering scheme (current values in `edge_dress`, next
values in `edge_dress_next`, swapped after each iteration) eliminates
all write conflicts and makes the iteration deterministic regardless
of execution order.
