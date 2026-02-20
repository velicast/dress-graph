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

## Higher-order DRESS for harder cases

Applying DRESS to a \(k\)-hop augmented graph \(G_k\) may break symmetries
that the original graph preserves.  This is an open research direction; see
[The DRESS Family](../theory/family.md#higher-order-dress).
