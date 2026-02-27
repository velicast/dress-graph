# ∇^k-DRESS: Higher-Order Refinement

∇^k-DRESS generalizes DRESS by applying vertex individualization at depth $k$, systematically climbing the Weisfeiler–Leman (WL) hierarchy.

## Definition

For a graph $G = (V, E)$, a DRESS variant $\mathcal{F}$, and a depth $k \ge 0$:

$$
\nabla^k\text{-DRESS}(\mathcal{F}, G) = h\!\left(\bigsqcup_{\substack{S \subset V,\; |S|=k}} \mathcal{F}(G^S)\right)
$$

where $G^S$ is the graph $G$ with the vertices in $S$ individualized (marked with distinct weights), $\bigsqcup$ denotes multiset union, and $h$ maps the pooled edge values into a histogram.

At depth $k$, the operator computes DRESS on all $\binom{n}{k}$ subproblems obtained by individualizing exactly $k$ vertices. Every individual edge similarity value from every subproblem is accumulated into a single histogram. Two graphs are compared by checking equality of their histograms.

Special cases:

- **$\nabla^0$:** Original-DRESS (no individualization).
- **$\nabla^1$:** Individualizes one vertex at a time — runs DRESS on each single-vertex-marked graph.

## Vertex Individualization

Unlike $\Delta^k$-DRESS, which removes vertices (reducing the graph), $\nabla^k$-DRESS **individualizes** them: vertices in $S$ receive distinct edge weights that break symmetry while preserving the full graph structure. This is more powerful because no structural information is lost — the entire graph remains available to the fixed-point iteration, but the marked vertices are forced to be distinguishable.

Concretely, for a set $S = \{v_1, \ldots, v_k\}$, the edges incident to each $v_i$ are reweighted with a unique marker, making the vertices in $S$ structurally non-equivalent even if they were automorphically equivalent in $G$.

## Histogram Representation

The graph fingerprint is represented as a histogram $h(d^*)$, which uniquely identifies the multiset of converged edge values. Since DRESS values are bounded in $[0, 2]$ and convergence tolerance is $\epsilon$, each edge value maps to one of $\lceil 2/\epsilon \rceil$ integer bins (e.g., $2 \times 10^{6}$ bins for $\epsilon = 10^{-6}$).

The memory footprint is constant regardless of $k$ — a single fixed-size integer array — whereas storing the raw multiset of $\binom{n}{k} \cdot |E|$ floating-point values would be prohibitive.

## Complexity

$$
\mathcal{O}\!\left(\binom{n}{k} \cdot I \cdot m \cdot d_{\max}\right)
$$

where $n = |V|$, $m = |E|$, $d_{\max}$ is the maximum degree, and $I$ is the number of DRESS iterations per subproblem. The $\binom{n}{k}$ subproblem computations are entirely independent and embarrassingly parallel. For fixed $k$, the cost is polynomial in $n$.

**Comparison with $(k{+}2)$-WL:** The cost of $(k{+}2)$-WL is $\mathcal{O}(n^{k+3})$ per iteration.
Space complexity is $\mathcal{O}(n + m)$ for ∇^k-DRESS, compared to $\mathcal{O}(n^{k+2})$ for $(k{+}2)$-WL.

## Provable Expressiveness

∇^k-DRESS is **provably at least as powerful as $(k{+}2)$-WL**:

| Depth $k$ | Expressiveness | WL equivalent |
|---|---|---|
| 0 | Original-DRESS | $\ge$ 2-WL |
| 1 | $\nabla^1$-DRESS | $\ge$ 3-WL |
| 2 | $\nabla^2$-DRESS | $\ge$ 4-WL |
| 3 | $\nabla^3$-DRESS | $\ge$ 5-WL |
| $k$ | $\nabla^k$-DRESS | $\ge$ $(k{+}2)$-WL |

Each level adds one WL dimension of expressiveness, with Original-DRESS contributing $\ge$ 2-WL on its own.

## CFI Benchmark Results

The Cai–Furer–Immerman (CFI) construction produces, for any base graph $G$, a pair of non-isomorphic graphs that $k$-WL cannot distinguish whenever $k$ is below the treewidth of $G$. For the complete graph $K_n$, the treewidth is $n - 1$, so CFI($K_n$) requires $(n{-}1)$-WL.

Results using Original-DRESS as the variant $\mathcal{F}$, with $\epsilon = 10^{-6}$ and a maximum of 100 iterations:

| Base graph | $\|V_{\text{CFI}}\|$ | WL req. | $\nabla^0$ | $\nabla^1$ | $\nabla^2$ | $\nabla^3$ |
|---|---|---|---|---|---|---|
| $K_3$ | 6 | 2-WL | ✓ | ✓ | ✓ | ✓ |
| $K_4$ | 16 | 3-WL | ✗ | ✓ | ✓ | ✓ |
| $K_5$ | 40 | 4-WL | ✗ | ✗ | ✓ | ✓ |
| $K_6$ | 96 | 5-WL | ✗ | ✗ | ✗ | ✓ |
| $K_7$ | 224 | 6-WL | ✗ | ✗ | ✗ | ✗ |
| $K_8$ | 512 | 7-WL | ✗ | ✗ | — | — |
| $K_9$ | 1152 | 8-WL | ✗ | ✗ | — | — |
| $K_{10}$ | 2560 | 9-WL | ✗ | ✗ | — | — |

✓ = pair distinguished, ✗ = pair not distinguished, — = not executed due to time constraints.

## ∇ vs Δ: Individualization vs Deletion

| Aspect | ∇^k-DRESS (individualization) | Δ^k-DRESS (deletion) |
|--------|-------------------------------|----------------------|
| Graph modification | Reweight edges of marked vertices | Remove vertices and incident edges |
| Graph size | Preserved (full $n$ vertices) | Reduced ($n - k$ vertices) |
| Information loss | None | Edges incident to deleted vertices lost |
| Expressiveness | $\ge (k{+}2)$-WL (proven) | Empirically matches $(k{+}2)$-WL |
| Practical advantage | Stronger theoretical guarantee | Faster per subproblem (smaller graphs) |

Both operators are embarrassingly parallel and produce identical-format histograms.

## Relationship to Subgraph GNNs

Methods such as ESAN and GNN-AK+ also use node-deleted or node-marked subgraphs to boost expressiveness. However, these are supervised methods that learn aggregation functions from data. ∇^k-DRESS is entirely unsupervised: the aggregation is the deterministic DRESS fixed point, and the histogram comparison is parameter-free. This makes it a canonical baseline for the expressiveness of subgraph-based approaches.

## Open Questions

1. **Sharp upper bound.** Is $\nabla^k$-DRESS **exactly** $(k{+}2)$-WL, or strictly more powerful for some graph families?
2. **Combining with other DRESS variants.** Whether a stronger base variant $\mathcal{F}$ (e.g., Motif-DRESS) shifts the expressiveness ladder upward.
3. **Optimal weight scheme.** The individualization weight assignment affects practical performance — finding the optimal scheme is an open problem.
