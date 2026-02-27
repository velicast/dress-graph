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
   out-neighborhoods.
2. **B-DRESS** (BACKWARD variant): uses \(\bar{w}(u,v) = w(v,u)\) and
   in-neighborhoods.

Each edge \((u,v)\) is now represented by a **pair** of values
\((f_{uv},\; b_{uv})\).  Sorting these pairs lexicographically produces a
directed fingerprint that distinguishes \(G\) from \(G^\top\) whenever their
structures differ.

**Why this works.**  F-DRESS captures the forward flow of similarity (how an
edge relates to its source's outgoing neighborhood), while B-DRESS captures
the backward flow (the target's incoming neighborhood).  A transpose swaps
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
  (same degree, same common-neighbor counts).  DRESS assigns the same value to
  every edge and cannot distinguish non-isomorphic SRGs with the same
  parameters.

## Relationship to Weisfeiler–Leman

DRESS is a **continuous relaxation of 1-WL** (color refinement).  Both
algorithms iterate over the same local structure — each node's
neighborhood — and converge to a fixed point.  Where 1-WL hashes
neighbor multisets into discrete colors, DRESS computes a cosine-like
ratio that yields continuous real-valued edge scores.

This has three practical consequences:

1. **Metric output.**  1-WL says "same or different"; DRESS says "how
   similar."  Every binary same/different test becomes a similarity
   query, and every color histogram becomes a real-valued distribution.
2. **Edge granularity.**  1-WL assigns one color per node; DRESS assigns
   one value per edge, giving a strictly finer structural fingerprint.
3. **Downstream utility.**  Continuous values can be thresholded, ranked,
   clustered, or fed directly into ML pipelines — none of which is possible
   with a discrete partition.

DRESS achieves 100 % accuracy on standard isomorphism benchmarks (MiVIA, IsoBench).
Original-DRESS **distinguishes beyond 1-WL**: it
[distinguishes the prism graph from \(K_{3,3}\)](#dress-distinguishes-graphs-that-1-wl-cannot),
a pair that 1-WL provably cannot separate
(see [Theorem 1 in the DRESS paper](https://github.com/velicast/dress-graph/blob/main/research/k-DRESS.pdf)).
It still fails on CFI constructions and strongly regular graphs with
identical parameters.
See [Properties — WL comparison](../theory/properties.md#weisfeilerleman-wl-color-refinement)
for a detailed side-by-side table.

### DRESS distinguishes graphs that 1-WL cannot

The **prism graph** (\(C_3 \square K_2\)) and \(K_{3,3}\) are both
3-regular on 6 nodes with 9 edges.  WL-1 assigns all nodes the same color
in both graphs (every node has degree 3 and all neighbor multisets are
identical), so it cannot distinguish them.

DRESS succeeds because it operates on **edges**, not nodes.  In the prism,
triangle edges share a common neighbor while matching edges do not;
these structurally different roles produce different DRESS values.
In \(K_{3,3}\), no two adjacent nodes share a common neighbor, so all
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
value \(d^*\).  Triangle edges have 1 common neighbor; their update
equation gives \(d^* = (4 + 4d^*) / (2 + 3d^*)\).  Matching edges have
0 common neighbors; their equation gives \(d^* = (4 + 2d^*) / (2 + 3d^*)\).
These yield \(d^* = (1+\sqrt{13})/3 \approx 1.535\) and \(d^* = 2/\sqrt{3} \approx 1.155\) respectively — a
contradiction.  Therefore the prism must have at least two distinct edge
values, while \(K_{3,3}\) (edge-transitive, 0 common neighbors everywhere)
has a single uniform value.  The sorted vectors necessarily differ.
\(\square\)

### DRESS reveals edge roles in regular graphs

On any \(d\)-regular graph WL-1 assigns a single color to every vertex and
a single color to every edge — it learns nothing.  DRESS, working at the
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
| Distance-1 (e.g. 0–1) | 1.549 | 10 | Share 2 common neighbors (most triangles) |
| Distance-2 (e.g. 0–2) | 1.166 | 10 | Share 1 common neighbor |
| Antipodal (e.g. 0–5) | 0.657 | 5 | Share 0 common neighbors (no triangles) |

All 10 node DRESS values are identical (\(\approx 4.022\)), which is
expected: the graph is vertex-transitive.  But the edges carry three
different roles that WL-1 is completely blind to.

### DRESS also has limits: strongly regular graphs

The **4×4 rook graph** and the **Shrikhande graph** are both
SRG(16, 6, 2, 2) — 6-regular on 16 vertices where every pair of adjacent
vertices shares exactly 2 common neighbors, and every pair of non-adjacent
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
| 4×4 Rook | 1.000 | — | 1 |
| Shrikhande | 1.000 | — | 1 |

DRESS cannot distinguish them.  Both graphs are edge-transitive, and every
edge has exactly the same local structure (2 common neighbors).  The DRESS
update equation sees identical inputs at every edge in both graphs, so it
converges to the same fixed point.

This confirms that Original-DRESS (triangle motif), like WL-1, fails on strongly regular graphs where
all edges are structurally indistinguishable.

## Higher-order DRESS for harder cases

The [DRESS paper](https://github.com/velicast/dress-graph/blob/main/research/k-DRESS.pdf) introduces Motif-DRESS and Δ-DRESS, and the [WL hierarchy paper](https://github.com/velicast/dress-graph/blob/main/research/vertex-k-DRESS.pdf) introduces ∇^k-DRESS as the primary higher-order variant.

### Motif-DRESS (K4 clique)

Motif-DRESS generalizes the neighborhood operator from triangles to arbitrary motifs. Using $K_4$ cliques as the motif, the neighborhood sizes differ between graphs even when triangle counts are identical. All experiments below use the $K_4$ clique motif.

| Pair | Result | Notes |
|------|--------|-------|
| Rook vs Shrikhande | **PASS** | Rook contains $K_4$ cliques; Shrikhande does not |
| T(8) vs Chang-1 | **PASS** | |
| T(8) vs Chang-2 | **PASS** | |
| T(8) vs Chang-3 | **PASS** | |
| Chang-1 vs Chang-2 | **PASS** | |
| Chang-1 vs Chang-3 | **PASS** | |
| Chang-2 vs Chang-3 | **PASS** | |

The specific SRG pairs tested above are known to be indistinguishable by 3-WL; each successful distinction therefore demonstrates that Motif-DRESS empirically exceeds 3-WL on these instances.

### Δ-DRESS

Δ-DRESS breaks symmetry by running standard DRESS on each node-deleted subgraph $G \setminus \{v\}$ for every $v \in V$. The graph fingerprint is the sorted multiset of $n$ converged edge-value vectors (one per deletion), compared by flattening and sorting without any summarization. Unlike approaches that modify the DRESS iteration itself, Δ-DRESS runs *unmodified* DRESS on structurally altered graphs. The multiset of deletion fingerprints is directly analogous to the *deck* in the Kelly–Ulam reconstruction conjecture. All experiments below use sorted multiset comparison (no fingerprint summarization).

| Graph / Pair | Result | Notes |
|--------------|--------|-------|
| Rook vs Shrikhande | **PASS** | SRG(16, 6, 2, 2) |
| $2 \times C_4$ vs $C_8$ | **PASS** | Both 2-regular on 8 nodes |
| Petersen vs Pentagonal Prism | **PASS** | Both 3-regular on 10 nodes |
| T(8) vs Chang-1 | **PASS** | |
| T(8) vs Chang-2 | **PASS** | |
| T(8) vs Chang-3 | **PASS** | |
| Chang-1 vs Chang-2 | **PASS** | |
| Chang-1 vs Chang-3 | **PASS** | |
| Chang-2 vs Chang-3 | **PASS** | |

### ∇^k-DRESS (Higher-Order Refinement)

∇^k-DRESS individualizes $k$ vertices at a time (reweighting their edges to break symmetry) instead of deleting them, preserving the full graph structure. This is provably at least as powerful as $(k{+}2)$-WL. See [the full ∇^k-DRESS treatment](../theory/nabla-k-dress.md) for CFI benchmark results and complexity analysis.

See also [Δ^k-DRESS (Iterated Deletion)](../theory/delta-ell-dress.md) for the generalization of Δ-DRESS to arbitrary depth.