# The Self-Loop Contribution

The self-loop in DRESS is not a numerical trick. It is mathematically
essential.

## Closed vs open neighbourhood

By defining \(N[u] = N(u) \cup \{u\}\) (closed neighbourhood), every node
becomes its own neighbour with a virtual self-edge:

- **Original weight:** \(w(u, u) = 1\)
- **Combined weight:** \(\bar{w}(u, u) = 2\) (since the self-loop is always
  seen from both sides, regardless of variant)
- **Similarity:** \(d_{uu} = 2\) (self-similarity)

## Why removal breaks the formula

Without self-loops, three critical failures occur:

### 1. Triangle-free graphs collapse

Stars, trees, paths, bipartite graphs: \(N(u) \cap N(v) = \emptyset\) for
every edge.  The numerator is zero.  Every edge gets \(d_{uv} = 0\).  DRESS
becomes useless on any graph without triangles.

### 2. The edge loses self-reference

Without self-loops, \(d_{uv}\) does not appear on the right-hand side of its
own equation.  The formula degenerates from a true fixed-point equation into a
simple ratio, losing the recursive, self-consistent character that gives DRESS
its power.

### 3. The denominator can be zero

If a node has no neighbours with nonzero dress values, \(\|u\| = 0\).  The
self-loop guarantees \(\|u\| \ge \sqrt{\bar{w}_{uu}\,d_{uu}} = \sqrt{4} = 2 > 0\) always.

## The self-loop makes DRESS a fixed-point equation

With self-loops, \(d_{uv}\) appears on both sides:

\[
d_{uv} = \frac{A(d) + c \cdot d_{uv}}
  {\sqrt{B(d) + c \cdot d_{uv}} \;\cdot\; \sqrt{C(d) + c \cdot d_{uv}}}
\]

This self-referential structure is precisely what creates the nonlinear coupled
system whose unique fixed point encodes the graph's structural information.
