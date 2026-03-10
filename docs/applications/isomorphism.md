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
  parameters.  **Δ¹-DRESS overcomes this**: all 51,816 tested graphs across 34 hard
  benchmark families are fully separated. See [Large-scale SRG separation](#large-scale-srg-separation) below.

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
identical parameters, but higher-order Δ^k-DRESS overcomes both; see
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

The [DRESS paper](https://github.com/velicast/dress-graph/blob/main/research/k-DRESS.pdf) introduces Motif-DRESS, Δ-DRESS, and Δ^k-DRESS as higher-order variants.

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
| $K_8$ | 512 | 7-WL | ✗ | ✗ | n/a | n/a |
| $K_9$ | 1,152 | 8-WL | ✗ | ✗ | n/a | n/a |
| $K_{10}$ | 2,560 | 9-WL | ✗ | ✗ | n/a | n/a |

The pattern is exact: **Δ^k-DRESS matches $(k{+}2)$-WL**. Each
additional deletion level adds one WL dimension. The boundary is sharp:
Δ³-DRESS distinguishes CFI($K_6$) (requires 5-WL) but fails on
CFI($K_7$) (requires 6-WL). "n/a" entries were not executed due to
time constraints.

*Summary of WL level per deletion depth:*

| Deletion depth $k$ | Max WL matched | Effective WL |
|:------------------:|:--------------:|:------------:|
| 0 | 2-WL | $k + 2$ |
| 1 | 3-WL | $k + 2$ |
| 2 | 4-WL | $k + 2$ |
| 3 | 5-WL | $k + 2$ |

The computational cost is $\mathcal{O}\bigl(\binom{n}{k} \cdot I \cdot m \cdot d_{\max}\bigr)$
, polynomial in $n$ for fixed $k$, while the equivalent
$(k{+}2)$-WL costs $\mathcal{O}(n^{k+3})$.

See the [DRESS paper](https://github.com/velicast/dress-graph/blob/main/research/k-DRESS.pdf)
for the full proofs and discussion.

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

To test Δ^1-DRESS beyond isolated pairs, we ran it on all known hard benchmark families.
These include the complete [Spence SRG collection](https://www.maths.gla.ac.uk/~es/srgraphs.php) (12 families, 43,703 graphs on up to 64 vertices), four additional SRG families from [McKay's collections](https://users.cecs.anu.edu.au/~bdm/data/graphs.html), and 18 constructed hard families (Miyazaki, Chang, Paley, Latin square, Steiner, and others).
All SRGs confound 2-WL by construction: identical degree, λ, μ, and spectrum within each family.
Plain DRESS (Δ^0) maps every graph in each SRG family to a single uniform edge value: **zero separation**.

| Family | Graphs | Unique | Pairs | Result |
|--------|:------:|:------:|------:|:------:|
| SRG(25,12,5,6) | 15 | 15 | 105 | **100 %** |
| SRG(26,10,3,4) | 10 | 10 | 45 | **100 %** |
| SRG(28,12,6,4) | 4 | 4 | 6 | **100 %** |
| SRG(29,14,6,7) | 41 | 41 | 820 | **100 %** |
| SRG(35,18,9,9) | 3,854 | 3,854 | 7,424,731 | **100 %** |
| SRG(36,14,4,6) | 180 | 180 | 16,110 | **100 %** |
| SRG(36,15,6,6) | 32,548 | 32,548 | 529,669,878 | **100 %** |
| SRG(37,18,8,9) | 6,760 | 6,760 | 22,845,420 | **100 %** |
| SRG(40,12,2,4) | 28 | 28 | 378 | **100 %** |
| SRG(45,12,3,3) | 78 | 78 | 3,003 | **100 %** |
| SRG(50,21,8,9) | 18 | 18 | 153 | **100 %** |
| SRG(64,18,2,6) | 167 | 167 | 13,861 | **100 %** |
| **Spence subtotal** | **43,703** | **43,703** | **559,974,510** | **100 %** |
| SRG(45,22,10,11) | 6 | 6 | 15 | **100 %** |
| SRG(63,32,16,26)-S | 4,466 | 4,466 | 9,970,345 | **100 %** |
| SRG(63,32,16,26)-Q | 3,511 | 3,511 | 6,161,805 | **100 %** |
| SRG(65,32,15,16) | 32 | 32 | 496 | **100 %** |
| **SRG total** | **51,718** | **51,718** | **576,107,171** | **100 %** |
| Constructed hard families (18) | 102 | 102 | n/a | **100 %** |
| **Grand total (distinct)** | **51,816** | **51,816** | **576,107,171** | **100 %** |

**All 51,816 graphs across 34 hard benchmark families are pairwise distinguished by Δ^1-DRESS**, resolving over 576 million within-family non-isomorphic pairs.

Δ¹-DRESS is strictly more powerful than 3-WL: the Rook L₂(4) vs. Shrikhande pair SRG(16,6,2,2), known to defeat 3-WL, is separated.
This places Δ¹-DRESS strictly above 3-WL; whether it is bounded above by 4-WL (≡ 3-FWL) remains open.

The fingerprint combines a pooled histogram with a multiplicity signature (an integer invariant counting repeated rows in the deleted-subgraph DRESS matrix), which resolved a single histogram collision in SRG(40,12,2,4).
Separation is stable across all rounding precisions from 6 to 14 decimal digits.

SRG data from [Spence](https://www.maths.gla.ac.uk/~es/srgraphs.php) and [McKay](https://users.cecs.anu.edu.au/~bdm/data/graphs.html). For the full paper: [One Deletion Suffices](https://github.com/velicast/dress-graph/blob/main/research/delta1-dress-hard-families.pdf).

See also [Δ^k-DRESS (Iterated Deletion)](../theory/delta-ell-dress.md) for the generalization of Δ-DRESS to arbitrary depth.