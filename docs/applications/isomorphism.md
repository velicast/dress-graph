# Graph Isomorphism Testing

## How it works

Isomorphic graphs have identical structure, so DRESS, which depends only on
structure, produces the same multiset of edge values for both.  Sorting the
DRESS vectors and comparing gives a fast isomorphism test.

### Undirected graphs

Run DRESS (UNDIRECTED variant).  Sort the resulting edge values.  Two graphs
are isomorphic only if their sorted vectors match.

### Directed graphs

For undirected graphs a single DRESS run suffices, but directed graphs require
more care.  The DIRECTED variant uses the combined weight
\(\bar{w}(u,v) = w(u,v) + w(v,u)\), which is symmetric in \(u\) and \(v\).
This means a graph \(G\) and its transpose \(G^\top\) (all arcs reversed)
can produce the same D-DRESS fingerprint.  Star graphs are a concrete example:
the hub-to-leaf and leaf-to-hub arcs swap roles under transposition, but the
symmetric combined weight erases the difference.

The solution is to run **two** passes:

1. **F-DRESS** (FORWARD variant): uses \(\bar{w}(u,v) = w(u,v)\) and
   out-neighbourhoods.
2. **B-DRESS** (BACKWARD variant): uses \(\bar{w}(u,v) = w(v,u)\) and
   in-neighbourhoods.

Each edge \((u,v)\) is now represented by a **pair** of values
\((f_{uv},\; b_{uv})\).  Sorting these pairs lexicographically produces a
directed fingerprint that distinguishes \(G\) from \(G^\top\) whenever their
structures differ.

**Why this works.**  F-DRESS captures the forward flow of similarity (how an
edge relates to its source's outgoing neighbourhood), while B-DRESS captures
the backward flow (the target's incoming neighbourhood).  A transpose swaps
the two components of every pair, so unless the graph is self-transpose, the
sorted pair vectors will differ.  This breaks exactly the symmetric-transpose
ambiguity that the single D-DRESS fingerprint misses.

## Results

| Benchmark | Accuracy |
|-----------|----------|
| MiVIA database | 100 % |
| IsoBench | 100 % |

## Limitations

DRESS provides a **necessary condition**, not a sufficient one.  Non-isomorphic
graphs *can* produce identical DRESS vectors:

- **CFI graphs** (Cai–Fürer–Immerman): constructed to defeat the
  Weisfeiler–Leman hierarchy.  DRESS fails these, as expected for any
  polynomial-time local method.
- **Strongly Regular Graphs (SRG)**: every edge has identical local structure
  (same degree, same common-neighbour counts).  DRESS assigns the same value to
  every edge and cannot distinguish non-isomorphic SRGs with the same
  parameters.

## Relationship to Weisfeiler–Leman

DRESS is a **continuous relaxation of 1-WL** (colour refinement).  Both
algorithms iterate over the same local structure — each node's
neighbourhood — and converge to a fixed point.  Where 1-WL hashes
neighbour multisets into discrete colours, DRESS computes a cosine-like
ratio that yields continuous real-valued edge scores.

This has three practical consequences:

1. **Metric output.**  1-WL says "same or different"; DRESS says "how
   similar."  Every binary same/different test becomes a similarity
   query, and every colour histogram becomes a real-valued distribution.
2. **Edge granularity.**  1-WL assigns one colour per node; DRESS assigns
   one value per edge, giving a strictly finer structural fingerprint.
3. **Downstream utility.**  Continuous values can be thresholded, ranked,
   clustered, or fed directly into ML pipelines — none of which is possible
   with a discrete partition.

DRESS achieves 100 % accuracy on standard isomorphism benchmarks.
Empirically it appears to share the same ceiling as 1-WL — it fails on
CFI constructions and strongly regular graphs with identical parameters —
but a formal proof that DRESS is at least as powerful as 1-WL is still
open.
See [Properties — WL comparison](../theory/properties.md#weisfeilerleman-wl-colour-refinement)
for a detailed side-by-side table.

### DRESS distinguishes graphs that 1-WL cannot

The **prism graph** (\(C_3 \square K_2\)) and \(K_{3,3}\) are both
3-regular on 6 nodes with 9 edges.  WL-1 assigns all nodes the same colour
in both graphs (every node has degree 3 and all neighbour multisets are
identical), so it cannot distinguish them.

DRESS succeeds because it operates on **edges**, not nodes.  In the prism,
triangle edges share a common neighbour while matching edges do not;
these structurally different roles produce different DRESS values.
In \(K_{3,3}\), no two adjacent nodes share a common neighbour, so all
edges are structurally equivalent.

```python
from dress import dress_fit

# Prism: C_3 □ K_2
r1 = dress_fit(6,
    [0, 1, 0, 3, 4, 3, 0, 1, 2],   # sources
    [1, 2, 2, 4, 5, 5, 3, 4, 5],   # targets
    max_iterations=100, epsilon=1e-10)

# K_{3,3}
r2 = dress_fit(6,
    [0, 0, 0, 1, 1, 1, 2, 2, 2],
    [3, 4, 5, 3, 4, 5, 3, 4, 5],
    max_iterations=100, epsilon=1e-10)

print(sorted(r1.edge_dress))   # two distinct values
print(sorted(r2.edge_dress))   # one uniform value
```

| Graph | Sorted DRESS vector |
|-------|---------------------|
| Prism | \([0.922, 0.922, 0.922, 1.709, 1.709, 1.709, 1.709, 1.709, 1.709]\) |
| \(K_{3,3}\) | \([1.155, 1.155, 1.155, 1.155, 1.155, 1.155, 1.155, 1.155, 1.155]\) |

The vectors differ, so DRESS correctly identifies the graphs as
non-isomorphic.  This is not an empirical accident: the proof follows
from the structure of the DRESS equation.

**Proof sketch.**  Suppose all 9 edges in the prism converge to the same
value \(d^*\).  Triangle edges have 1 common neighbour; their update
equation gives \(d^* = (8 + 4d^*) / (4 + 3d^*)\).  Matching edges have
0 common neighbours; their equation gives \(d^* = (8 + 2d^*) / (4 + 3d^*)\).
These yield \(d^* = \sqrt{8/3}\) and \(d^* = 4/3\) respectively — a
contradiction.  Therefore the prism must have at least two distinct edge
values, while \(K_{3,3}\) (edge-transitive, 0 common neighbours everywhere)
has a single uniform value.  The sorted vectors necessarily differ.
\(\square\)

### DRESS reveals edge roles in regular graphs

On any \(d\)-regular graph WL-1 assigns a single colour to every vertex and
a single colour to every edge — it learns nothing.  DRESS, working at the
edge level, can still expose structurally distinct edge roles.

**Example: circulant graph \(C(10,\{1,2,5\})\).**  This is a 5-regular
graph on 10 vertices where each vertex \(i\) connects to
\(i \pm 1\), \(i \pm 2\), and \(i + 5\) (mod 10).

```python
from dress import dress_fit

n = 10
edges = set()
for i in range(n):
    for s in [1, 2]:
        edges.add((min(i, (i+s)%n), max(i, (i+s)%n)))
        edges.add((min(i, (i-s)%n), max(i, (i-s)%n)))
    edges.add((min(i, (i+5)%n), max(i, (i+5)%n)))

edges = sorted(edges)
r = dress_fit(n, [u for u,v in edges], [v for u,v in edges])
print(f"Iterations: {r.iterations}")
for s, t, d in zip(r.sources, r.targets, r.edge_dress):
    print(f"  ({s},{t})  {d:.6f}")
```

DRESS converges in 12 iterations and produces **three distinct edge
values**:

| Edge type | DRESS value | Count | Structural meaning |
|-----------|-------------|-------|--------------------|
| Distance-1 (e.g. 0–1) | 1.549 | 10 | Share 2 common neighbours (most triangles) |
| Distance-2 (e.g. 0–2) | 1.166 | 10 | Share 1 common neighbour |
| Antipodal (e.g. 0–5) | 0.657 | 5 | Share 0 common neighbours (no triangles) |

All 10 node DRESS values are identical (\(\approx 4.022\)), which is
expected: the graph is vertex-transitive.  But the edges carry three
different roles that WL-1 is completely blind to.

### DRESS also has limits: strongly regular graphs

The **4×4 rook graph** and the **Shrikhande graph** are both
SRG(16, 6, 2, 2) — 6-regular on 16 vertices where every pair of adjacent
vertices shares exactly 2 common neighbours, and every pair of non-adjacent
vertices also shares exactly 2.

```python
from dress import dress_fit

# 4×4 rook graph: vertex (r,c) connects to all in same row/column
n, edges = 16, set()
for r in range(4):
    for c in range(4):
        v = r*4 + c
        for c2 in range(4):
            if c2 != c: edges.add((min(v, r*4+c2), max(v, r*4+c2)))
        for r2 in range(4):
            if r2 != r: edges.add((min(v, r2*4+c), max(v, r2*4+c)))
edges = sorted(edges)
rook = dress_fit(n, [u for u,v in edges], [v for u,v in edges])

# Shrikhande graph: Cayley graph of Z4×Z4, generators {(0,1),(1,0),(1,1)}
edges = set()
gens = [(0,1),(1,0),(1,1),(0,3),(3,0),(3,3)]
for r in range(4):
    for c in range(4):
        v = r*4 + c
        for dr, dc in gens:
            u = ((r+dr)%4)*4 + (c+dc)%4
            edges.add((min(v,u), max(v,u)))
edges = sorted(edges)
shri = dress_fit(n, [u for u,v in edges], [v for u,v in edges])

print(f"Rook       edge DRESS: {rook.edge_dress[0]:.6f} (uniform)")
print(f"Shrikhande edge DRESS: {shri.edge_dress[0]:.6f} (uniform)")
```

Both graphs produce **identical, uniform** DRESS values:

| | Edge DRESS | Node DRESS | Distinct edge values |
|---|---|---|---|
| 4×4 Rook | 1.215 | 4.311 | 1 |
| Shrikhande | 1.215 | 4.311 | 1 |

DRESS cannot distinguish them.  Both graphs are edge-transitive, and every
edge has exactly the same local structure (2 common neighbours).  The DRESS
update equation sees identical inputs at every edge in both graphs, so it
converges to the same fixed point.

This confirms that DRESS, like WL-1, fails on strongly regular graphs where
all edges are structurally indistinguishable.

## Higher-order DRESS for harder cases

Applying DRESS to a \(k\)-hop augmented graph \(G_k\) may break symmetries
that the original graph preserves.  This is an open research direction; see
[The DRESS Family](../theory/family.md#higher-order-dress).
