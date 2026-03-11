# Δ^k-DRESS: Iterated Vertex Deletion

Δ^k-DRESS generalizes Δ-DRESS by applying $k$ levels of iterated vertex deletion, systematically climbing the Weisfeiler–Leman (WL) hierarchy.

## Definition

For a graph $G = (V, E)$, a DRESS variant $\mathcal{F}$, and a deletion depth $k \ge 0$:

$$
\Delta^k\text{-DRESS}(\mathcal{F}, G) = \{\!\{ \operatorname{sort}(\mathcal{F}(G \setminus S)) : S \subset V,\; |S| = k \}\!\}
$$

where $G \setminus S$ is the subgraph induced by $V \setminus S$ and $\mathcal{F}(G \setminus S)$ is the converged DRESS edge-value vector.

At depth $k$, the operator computes DRESS on all $\binom{n}{k}$ subgraphs obtained by removing exactly $k$ vertices. The fingerprint is the multiset of per-deletion sorted edge-value vectors. Two graphs are compared by checking equality of their fingerprint multisets.

Special cases:

- **$\Delta^0$:** Original-DRESS (no deletion).
- **$\Delta^1$:** Equivalent to Δ-DRESS — runs DRESS on each single-vertex-deleted subgraph.

## CFI Benchmark Results

The Cai–Furer–Immerman (CFI) construction produces, for any base graph $G$, a pair of non-isomorphic graphs that $k$-WL cannot distinguish whenever $k$ is below the treewidth of $G$. For the complete graph $K_n$, the treewidth is $n - 1$, so CFI($K_n$) requires $(n{-}1)$-WL.

Results using Original-DRESS as the variant $\mathcal{F}$, with $\epsilon = 10^{-6}$ and a maximum of 100 iterations:

| Base graph | $\|V_{\text{CFI}}\|$ | WL req. | $\Delta^0$ | $\Delta^1$ | $\Delta^2$ | $\Delta^3$ |
|---|---|---|---|---|---|---|
| $K_3$ | 6 | 2-WL | ✓ | ✓ | ✓ | ✓ |
| $K_4$ | 16 | 3-WL | ✗ | ✓ | ✓ | ✓ |
| $K_5$ | 40 | 4-WL | ✗ | ✗ | ✓ | ✓ |
| $K_6$ | 96 | 5-WL | ✗ | ✗ | ✗ | ✓ |
| $K_7$ | 224 | 6-WL | ✗ | ✗ | ✗ | ✗ |
| $K_8$ | 512 | 7-WL | ✗ | ✗ | - | - |
| $K_9$ | 1152 | 8-WL | ✗ | ✗ | - | - |
| $K_{10}$ | 2560 | 9-WL | ✗ | ✗ | - | - |

✓ = pair distinguished, ✗ = pair not distinguished, - = not executed due to time constraints.

## Relationship to Subgraph GNNs

Methods such as ESAN and GNN-AK+ also use vertex-deleted or vertex-marked subgraphs to boost expressiveness. However, these are supervised methods that learn aggregation functions from data. $\Delta^k$-DRESS is entirely unsupervised: the aggregation is the deterministic DRESS fixed point, and the fingerprint comparison is parameter-free. This makes it a canonical baseline for the expressiveness of subgraph-based approaches.

## Theoretical Results

The companion theoretical paper (*vertex-k-DRESS*) provides two levels of justification for the staircase pattern:

### CFI Staircase Theorem (Unconditional)

For all $k \geq 0$, $\Delta^k$-DRESS distinguishes CFI($K_{k+3}$) from CFI'($K_{k+3}$). This is proved by induction using two new results:

- **CFI Deck Separation Theorem.** For $n \geq 3$, no deletion card of CFI($K_n$) is isomorphic to any deletion card of CFI'($K_n$): the decks are completely disjoint. The CFI twist survives every single-vertex deletion.

- **Virtual Pebble Lemma.** For $n \geq 3$ and any vertex $w$, $(n{-}2)$-WL distinguishes CFI($K_n$) $\setminus \{w\}$ from CFI'($K_n$) $\setminus \{w\}$. The deleted vertex acts as a "free pebble": the damaged gadget is pinned by its unique size, reducing the pebble count by exactly one. This is the key "step down" that makes the induction tight — the damaged cards need exactly one WL level less than the undamaged pair.

**No conjectures are needed for the CFI proof.**

### General Theorem (Conditional)

For every $k \geq 0$, $\Delta^k$-DRESS $\geq$ $(k{+}2)$-WL for all graphs, conditional on a single structural conjecture:

**WL-Deck Separation Conjecture.** If $(j{+}1)$-WL distinguishes $G$ from $H$, then the multisets of $j$-WL stable colorings over their 1-decks differ.

This conjecture is independent of the Kelly–Ulam Reconstruction Conjecture, which is not needed anywhere in the proof. The CFI Staircase Theorem provides strong evidence for the conjecture: it holds unconditionally for all CFI($K_n$) instances.

## Open Questions

1. **Can the WL-Deck Separation Conjecture be proved?** A proof via the descriptive complexity of $C^{j+1}$ seems the most promising route.
2. **Are there non-CFI graph families harder for $\Delta^k$-DRESS?** The Virtual Pebble Lemma exploits special CFI structure; whether analogous structure exists for Miyazaki graphs or other hard families is open.
