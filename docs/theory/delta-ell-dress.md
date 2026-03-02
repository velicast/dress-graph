# Δ^k-DRESS: Iterated Node Deletion

Δ^k-DRESS generalizes Δ-DRESS by applying $k$ levels of iterated node deletion, systematically climbing the Weisfeiler–Leman (WL) hierarchy.

## Definition

For a graph $G = (V, E)$, a DRESS variant $\mathcal{F}$, and a deletion depth $k \ge 0$:

$$
\Delta^k\text{-DRESS}(\mathcal{F}, G) = h\!\left(\bigsqcup_{\substack{S \subset V,\; |S|=k}} \mathcal{F}(G \setminus S)\right)
$$

where $G \setminus S$ is the subgraph induced by $V \setminus S$, $\bigsqcup$ denotes multiset union, and $h$ maps the pooled edge values into a histogram.

At depth $k$, the operator computes DRESS on all $\binom{n}{k}$ subgraphs obtained by removing exactly $k$ vertices. Every individual edge similarity value from every subgraph is accumulated into a single histogram. Two graphs are compared by checking equality of their histograms.

Special cases:

- **$\Delta^0$:** Original-DRESS (no deletion).
- **$\Delta^1$:** Equivalent to Δ-DRESS - runs DRESS on each single-node-deleted subgraph.

## Histogram Representation

The graph fingerprint can be represented equivalently as the sorted vector $\text{sort}(d^*)$ or as the histogram $h(d^*)$; both uniquely identify the multiset of converged edge values. In the unweighted case DRESS values are bounded in $[0, 2]$ and convergence tolerance is $\epsilon$, so each edge value maps to one of $\lfloor 2/\epsilon \rfloor + 1$ integer bins (e.g., $2 \times 10^{6} + 1$ bins for $\epsilon = 10^{-6}$). For weighted graphs, where values may exceed $2$, the bin count grows to $\lfloor d_{\max}/\epsilon \rfloor + 1$ where $d_{\max}$ is an a priori upper bound computed from the edge weights.

Individual subgraph fingerprints are not recoverable from the pooled histogram. The memory footprint is constant regardless of $k$ - a single fixed-size integer array - whereas storing the raw multiset of $\binom{n}{k} \cdot |E|$ floating-point values would be prohibitive. This fixed-size representation, combined with the deletion operator, empirically matches the discriminative power of $k$-WL methods that require $\mathcal{O}(n^{k+1})$ storage.

## Complexity

$$
\mathcal{O}\!\left(\binom{n}{k} \cdot I \cdot m \cdot d_{\max}\right)
$$

where $n = |V|$, $m = |E|$, $d_{\max}$ is the maximum degree, and $I$ is the number of DRESS iterations per subgraph. The $\binom{n}{k}$ subgraph computations are entirely independent and embarrassingly parallel. For fixed $k$, the cost is polynomial in $n$.

**Comparison with $k$-WL:** The cost of $k$-WL is $\mathcal{O}(n^{k+1})$ per iteration. For example, to match 5-WL on CFI($K_6$), $\Delta^3$-DRESS costs $\mathcal{O}(n^3 \cdot m \cdot d_{\max})$ versus $\mathcal{O}(n^6)$ for 5-WL.

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

## The Staircase Pattern

The results reveal a strikingly regular pattern:

| Deletion depth $k$ | Max WL matched | Effective WL |
|---|---|---|
| 0 | 2-WL | $k + 2$ |
| 1 | 3-WL | $k + 2$ |
| 2 | 4-WL | $k + 2$ |
| 3 | 5-WL | $k + 2$ |

$\Delta^k$-DRESS empirically matches $(k+2)$-WL on the CFI family:

- **$\Delta^0$ (Original-DRESS)** distinguishes CFI($K_3$) (2-WL) but fails on all larger CFI pairs.
- **$\Delta^1$** extends the range to CFI($K_4$) (3-WL), consistent with Δ-DRESS distinguishing SRG pairs that confound 3-WL.
- **$\Delta^2$** reaches CFI($K_5$) (4-WL).
- **$\Delta^3$** reaches CFI($K_6$) (5-WL) but fails on CFI($K_7$) (6-WL), confirming the sharp boundary at $(k+2)$-WL. The CFI($K_7$) test required processing all $\binom{224}{3} = 1{,}848{,}224$ subgraphs (approximately 19.3 billion edge values).

Each deletion level adds one WL level of expressiveness, with Original-DRESS contributing 2-WL on its own. The computational cost grows as $\binom{n}{k}$ (polynomial for fixed $k$), while the equivalent $(k+2)$-WL cost grows as $n^{k+3}$.

## Relationship to Subgraph GNNs

Methods such as ESAN and GNN-AK+ also use node-deleted or node-marked subgraphs to boost expressiveness. However, these are supervised methods that learn aggregation functions from data. $\Delta^k$-DRESS is entirely unsupervised: the aggregation is the deterministic DRESS fixed point, and the histogram comparison is parameter-free. This makes it a canonical baseline for the expressiveness of subgraph-based approaches.

## Connection to the Reconstruction Conjecture

The Kelly–Ulam conjecture states that graphs with $n \ge 3$ are determined by their deck of node-deleted subgraphs. Conceptually, $\Delta^1$-DRESS computes a continuous relaxation of this deck: each "card" is the DRESS fingerprint of a node-deleted subgraph. In practice, all edge values are accumulated into a fixed-size histogram for memory efficiency, though individual cards are not recoverable.

## Fixed-Parameter Tractability

For fixed $k$, $\Delta^k$-DRESS runs in polynomial time. The empirical pattern $\Delta^k \approx (k+2)$-WL suggests that the minimum deletion depth needed to distinguish CFI($K_n$) grows as $n - 2$, which is linear in $n$. While each additional deletion level is polynomial, the number of required levels grows with the base graph size.

## Open Questions

1. **Does the staircase continue?** Testing $\Delta^4$ on CFI($K_7$) would confirm whether the pattern extends to $k = 4$.
2. **Is the pattern specific to CFI, or does it hold more broadly?** Other hard graph families may behave differently.
3. **Are there graph families where $\Delta^k$-DRESS fails for all constant $k$?** CFI graphs over graphs of unbounded treewidth would be natural candidates.
4. **Does combining $\Delta^k$ with other DRESS variants yield further gains?** Whether a stronger base variant $\mathcal{F}$ (e.g., Motif-DRESS) shifts the staircase upward is an open question.
