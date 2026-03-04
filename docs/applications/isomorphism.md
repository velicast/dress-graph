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

## Limitations of Original-DRESS (Δ⁰)

Original-DRESS provides a **necessary condition**, not a sufficient one.  Non-isomorphic
graphs *can* produce identical DRESS vectors:

- **CFI graphs** (Cai–Fürer–Immerman): constructed to defeat the
  Weisfeiler–Leman hierarchy.  Original-DRESS fails on CFI(\(K_4\)) and above.
  **Δ^k-DRESS overcomes this**: each deletion level adds one WL dimension,
  matching \((k{+}2)\)-WL. See [CFI Staircase](#cfi-staircase) below.
- **Strongly Regular Graphs (SRG)**: every edge has identical local structure
  (same degree, same common-neighbor counts).  Original-DRESS assigns the same value to
  every edge and cannot distinguish non-isomorphic SRGs with the same
  parameters.  **Δ¹-DRESS overcomes this**: all 7,983 tested SRGs are fully
  separated. See [Large-scale SRG separation](#large-scale-srg-separation) below.

## Relationship to Weisfeiler–Leman

DRESS **matches 2-WL** in expressiveness.  Where 2-WL assigns discrete
colors to node pairs, DRESS computes a cosine-like ratio that yields
continuous real-valued edge scores - achieving the same distinguishing
power at \(O(E)\) cost per iteration.

This has three practical consequences:

1. **Metric output.**  2-WL says "same or different"; DRESS says "how
   similar."  Every binary same/different test becomes a similarity
   query, and every color histogram becomes a real-valued distribution.
2. **Edge granularity.**  2-WL assigns one color per node pair; DRESS assigns
   one value per edge, giving a compact structural fingerprint.
3. **Downstream utility.**  Continuous values can be thresholded, ranked,
   clustered, or fed directly into ML pipelines - none of which is possible
   with a discrete partition.

DRESS achieves 100 % accuracy on standard isomorphism benchmarks (MiVIA, IsoBench).
Original-DRESS **matches 2-WL**: it
[distinguishes the prism graph from \(K_{3,3}\)](#dress-matches-2-wl),
a pair that 1-WL cannot separate but 2-WL can
(see [Theorem 1 in the DRESS paper](https://github.com/velicast/dress-graph/blob/main/research/k-DRESS.pdf)).
Original-DRESS fails on CFI constructions and strongly regular graphs with
identical parameters — but higher-order Δ^k-DRESS overcomes both; see
[CFI Staircase](#cfi-staircase) and [Large-scale SRG separation](#large-scale-srg-separation) below.
See [Properties - WL comparison](../theory/properties.md#weisfeilerleman-wl-color-refinement)
for a detailed side-by-side table.

### DRESS matches 2-WL

The **prism graph** (\(C_3 \square K_2\)) and \(K_{3,3}\) are both
3-regular on 6 nodes with 9 edges.  1-WL assigns all nodes the same color
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
These yield \(d^* = (1+\sqrt{13})/3 \approx 1.535\) and \(d^* = 2/\sqrt{3} \approx 1.155\) respectively - a
contradiction.  Therefore the prism must have at least two distinct edge
values, while \(K_{3,3}\) (edge-transitive, 0 common neighbors everywhere)
has a single uniform value.  The sorted vectors necessarily differ.
\(\square\)

### DRESS reveals edge roles in regular graphs

On any \(d\)-regular graph 2-WL assigns a single color to every vertex and
a single color to every edge - it learns nothing.  DRESS, working at the
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
different roles that 1-WL is completely blind to.

### DRESS also has limits: strongly regular graphs

The **4×4 rook graph** and the **Shrikhande graph** are both
SRG(16, 6, 2, 2) - 6-regular on 16 vertices where every pair of adjacent
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
| 4×4 Rook | 1.000 | - | 1 |
| Shrikhande | 1.000 | - | 1 |

DRESS cannot distinguish them.  Both graphs are edge-transitive, and every
edge has exactly the same local structure (2 common neighbors).  The DRESS
update equation sees identical inputs at every edge in both graphs, so it
converges to the same fixed point.

This confirms that Original-DRESS (triangle motif), like 1-WL, fails on strongly regular graphs where
all edges are structurally indistinguishable.

## Higher-order DRESS for harder cases

The [DRESS paper](https://github.com/velicast/dress-graph/blob/main/research/k-DRESS.pdf) introduces Motif-DRESS and Δ-DRESS, and the [WL hierarchy paper](https://github.com/velicast/dress-graph/blob/main/research/vertex-k-DRESS.pdf) introduces Δ^k-DRESS as the primary higher-order variant.

### CFI Staircase

The [Cai–Fürer–Immerman construction](https://en.wikipedia.org/wiki/Cai%E2%80%93F%C3%BCrer%E2%80%93Immerman_graph)
produces the canonical hard instances for the WL hierarchy: distinguishing
CFI($K_n$) from CFI'($K_n$) requires $(n{-}1)$-WL. Δ^k-DRESS
climbs the staircase one level per deletion depth:

| Base graph | Vertices | WL req. | Δ⁰ | Δ¹ | Δ² | Δ³ |
|:----------:|:--------:|:-------:|:---:|:---:|:---:|:---:|
| $K_3$ | 6 | 2-WL | ✓ | ✓ | ✓ | ✓ |
| $K_4$ | 16 | 3-WL | ✗ | ✓ | ✓ | ✓ |
| $K_5$ | 40 | 4-WL | ✗ | ✗ | ✓ | ✓ |
| $K_6$ | 96 | 5-WL | ✗ | ✗ | ✗ | ✓ |
| $K_7$ | 224 | 6-WL | ✗ | ✗ | ✗ | ✗ |
| $K_8$ | 512 | 7-WL | ✗ | ✗ | — | — |
| $K_9$ | 1,152 | 8-WL | ✗ | ✗ | — | — |
| $K_{10}$ | 2,560 | 9-WL | ✗ | ✗ | — | — |

The pattern is exact: **Δ^k-DRESS matches $(k{+}2)$-WL**. Each
additional deletion level adds one WL dimension. The boundary is sharp:
Δ³-DRESS distinguishes CFI($K_6$) (requires 5-WL) but fails on
CFI($K_7$) (requires 6-WL). "—" entries were not executed due to
time constraints.

*Summary of WL level per deletion depth:*

| Deletion depth $k$ | Max WL matched | Effective WL |
|:------------------:|:--------------:|:------------:|
| 0 | 2-WL | $k + 2$ |
| 1 | 3-WL | $k + 2$ |
| 2 | 4-WL | $k + 2$ |
| 3 | 5-WL | $k + 2$ |

The computational cost is $\mathcal{O}\bigl(\binom{n}{k} \cdot I \cdot m \cdot d_{\max}\bigr)$
— polynomial in $n$ for fixed $k$ — while the equivalent
$(k{+}2)$-WL costs $\mathcal{O}(n^{k+3})$.

See [Paper 2](https://github.com/velicast/dress-graph/blob/main/research/vertex-k-DRESS.pdf)
for the full proofs (Theorem 2: Δ^k-DRESS ≥ (k+2)-WL under the
Reconstruction Conjecture) and discussion.

### Motif-DRESS (K4 clique)

Motif-DRESS generalizes the neighborhood operator from triangles to arbitrary motifs. Using $K_4$ cliques as the motif, the neighborhood sizes differ between graphs even when triangle counts are identical. All experiments below use the $K_4$ clique motif.

| Pair | Result | Notes |
|------|--------|-------|
| Rook vs Shrikhande | **PASS** | Rook contains $K_4$ cliques; Shrikhande does not |
| T(8) vs Chang-1 | **PASS** | |
| T(8) vs Chang-2 | **PASS** | |
| T(8) vs Chang-3 | **PASS** | |
| Chang-1 vs Chang-2 | FAIL | All three Chang graphs have identical $K_4$-neighborhood structure per edge |
| Chang-1 vs Chang-3 | FAIL | |
| Chang-2 vs Chang-3 | FAIL | |

Motif-$K_4$ distinguishes 3 of the 6 Chang pairs (T(8) vs each Chang graph) and 1/1 Rook vs Shrikhande. The three Chang graphs are pairwise indistinguishable because they share identical $K_4$-neighborhood structure per edge. The specific SRG pairs tested above are known to be indistinguishable by 2-WL; each successful distinction therefore demonstrates that Motif-DRESS empirically exceeds 2-WL on these instances.

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

#### Large-scale SRG separation

To test Δ-DRESS (i.e. Δ^1-DRESS) beyond isolated pairs, we ran it on
**7,983 strongly regular graphs** from the repository of
[Krystal Guo](https://github.com/kguo-sagecode/Strongly-regular-graphs).
These graphs share SRG parameters within each family —
identical degree, λ, μ, spectrum — so they all confound 2-WL.
Plain DRESS (Δ^0) maps every graph in each family to a single
uniform edge value: **zero separation**.

| Family | Parameters | Graphs | Δ^1 unique | Separated | Min L∞ between closest pair |
|--------|-----------|:------:|:----------:|:---------:|:---------------------------:|
| Conference (Mathon) | (45, 22, 10, 11) | 6 | 6 | **100 %** | 4.16 × 10⁻³ |
| Steiner block S(2,4,28) | (63, 32, 16, 16) | 4,466 | 4,466 | **100 %** | 1.95 × 10⁻³ |
| Quasi-symmetric 2-designs | (63, 32, 16, 16) | 3,511 | 3,511 | **100 %** | 2.23 × 10⁻³ |

**All 7,983 graphs are pairwise distinguished by Δ^1-DRESS.**

The "Min L∞" column reports the smallest element-wise maximum difference
between the sorted fingerprints of any sampled graph pair (1,000 random
pairs per family). Values around 10⁻³ confirm that the separations are
genuine, not floating-point artifacts — this was further validated by
checking that the unique count remains **stable across all rounding
precisions** from 6 to 14 decimal digits.

Cross-file analysis between the two SRG(63, 32, 16, 16) sources
(Steiner and quasi-symmetric) found **zero collisions**, confirming that
the two constructions produce entirely disjoint isomorphism classes.

Computation used 32 CPU cores with the pure-Python DRESS backend:
~10 min for 4,466 graphs (281,358 DRESS runs) and ~8 min for 3,511
graphs (221,193 DRESS runs).

See also [Δ^k-DRESS (Iterated Deletion)](../theory/delta-ell-dress.md) for the generalization of Δ-DRESS to arbitrary depth.