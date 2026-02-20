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

DRESS achieves 100 % accuracy on standard isomorphism benchmarks, but
shares the same theoretical ceiling as 1-WL: it cannot distinguish CFI
constructions or strongly regular graphs with identical parameters.
See [Properties — WL comparison](../theory/properties.md#weisfeilerleman-wl-colour-refinement)
for a detailed side-by-side table.

## Higher-order DRESS for harder cases

Applying DRESS to a \(k\)-hop augmented graph \(G_k\) may break symmetries
that the original graph preserves.  This is an open research direction; see
[The DRESS Family](../theory/family.md#higher-order-dress).
