# Edge Robustness

## Motivation

DRESS assigns every edge a value reflecting its structural importance.
A natural question: **can these values double as an edge-importance
ranking for robustness analysis?**

If so, any DRESS computation - whether for
[classification](classification.md), [retrieval](retrieval.md), or
community detection - already contains a principled edge-removal ordering.
No additional algorithm is required.

We compare DRESS against established baselines *and* against **adaptive betweenness centrality**
- the theoretical ceiling that recomputes betweenness after every
single removal at \(O(n \cdot m^2)\) cost.

## Setup

### Progressive edge removal

For a connected graph \(G\), we progressively remove edges according to an
ordering and track the **Largest Connected Component (LCC) fraction** - the
fraction of nodes still reachable from the largest component.  The area under
this curve (AUC) summarizes how quickly the ordering fragments the graph:
**lower AUC = more damaging ordering**.

### Strategies compared

| Strategy | Type | Description | Complexity |
|----------|------|-------------|------------|
| **DRESS-asc** | static | Ascending standard DRESS values (weakest first) | \(O(k \cdot m)\) |
| DRESS-desc | static | Descending standard DRESS values | Same fit |
| Betweenness | static | Descending edge-betweenness centrality | \(O(n \cdot m)\) |
| CoreHD | static | Min core-number, ties by degree product (Zdeborová et al., 2016) | \(O(m)\) |
| CI | static | Collective Influence: \(\text{CI}(u) + \text{CI}(v)\), descending (Morone & Makse, 2015) | \(O(n \cdot d^\ell)\) |
| Fiedler | static | Spectral cut: \((x_u - x_v)^2\) from Fiedler vector (Fiedler, 1973) | \(O(m)\) |
| DegProduct | static | \(\deg(u) \cdot \deg(v)\) descending | \(O(m)\) |
| Random | static | Uniformly random order | \(O(m)\) |
| **BC-adaptive** | adaptive | Recompute betweenness after each removal | \(O(n \cdot m^2)\) |
| **DRESS-adaptive** | adaptive | Recompute DRESS after each removal | \(O(k \cdot m^2)\) |

Static strategies fix the ordering before removal begins.  Adaptive
strategies recompute after every removal - they are the theoretical
ceiling but orders of magnitude more expensive.

### Why ascending?

Edges with **low** DRESS values are structurally isolated: they lack common
neighbors and receive little support during convergence.  These are
precisely the bridges and bottlenecks whose removal disconnects components.
Removing them first is the DRESS-native attack strategy.

## Benchmark

### Datasets

We evaluate on **10 TU benchmark datasets** spanning molecular, biological,
and social network domains:

| Dataset | Domain | Graphs sampled |
|---------|--------|----------------|
| MUTAG | Molecular | 20 |
| PTC_MR | Molecular | 20 |
| PROTEINS | Biological | 20 |
| DD | Biological | 20 |
| ENZYMES | Biological | 20 |
| NCI1 | Molecular | 20 |
| IMDB-BINARY | Social | 20 |
| IMDB-MULTI | Social | 20 |
| REDDIT-BINARY | Social | 20 |
| REDDIT-MULTI-5K | Social | 20 |

For each dataset, 20 connected graphs with 15–300 nodes are sampled.
Total: **200 real graphs** plus 4 synthetic.

### Synthetic graphs

| Graph | \(n\) | \(m\) | Type |
|-------|-------|-------|------|
| BA-200 | 200 | 591 | Barabási–Albert (preferential attachment) |
| WS-200 | 200 | 600 | Watts–Strogatz (small-world) |
| Grid-15×15 | 225 | 420 | 2D lattice |
| StarCliques-8×10 | 81 | 288 | Hub connected to 8 cliques of size 10 |

## Results

### Overall ranking (200 real graphs, 10 datasets)

Mean AUC across all graphs; lower is better.

| Rank | Strategy | Mean AUC | Complexity | Type |
|------|----------|----------|------------|------|
| 1 | BC-adaptive | 0.256 | \(O(n \cdot m^2)\) | adaptive |
| 2 | DRESS-adaptive | 0.304 | \(O(k \cdot m^2)\) | adaptive |
| 3 | **DRESS-asc** | **0.348** | \(O(k \cdot m)\) | **static** |
| 4 | Betweenness | 0.373 | \(O(n \cdot m)\) | static |
| 5 | Fiedler | 0.376 | \(O(m)\) | static |
| 6 | DegProduct | 0.423 | \(O(m)\) | static |
| 7 | CoreHD | 0.439 | \(O(m)\) | static |
| 8 | CI | 0.478 | \(O(n \cdot d^\ell)\) | static |
| 9 | Random | 0.528 | - | baseline |

!!! success "Headline result"
    **DRESS-asc is the best static strategy overall**, ranking #3
    behind only the two adaptive methods that are orders of magnitude
    more expensive.

### Ceiling capture

DRESS-asc captures **66 %** of the maximum achievable improvement
over random:

\[
\frac{\text{AUC}_{\text{Random}} - \text{AUC}_{\text{DRESS-asc}}}
     {\text{AUC}_{\text{Random}} - \text{AUC}_{\text{BC-adaptive}}}
= \frac{0.528 - 0.348}{0.528 - 0.256} = 66\%
\]

This is achieved with a **static** precomputed ranking at \(O(km)\),
vs recomputing betweenness after every single removal at \(O(nm^2)\).

### DRESS-asc vs static baselines (200 graphs)

| Comparison | DRESS-asc wins | Win rate | Significant datasets |
|------------|-------------------|----------|----------------------|
| vs Betweenness | 158 / 200 | **79 %** | 7 / 10 |
| vs Fiedler | 156 / 200 | **78 %** | 7 / 10 |
| vs CoreHD | ~180 / 200 | **~90 %** | 9 / 10 |
| vs CI | ~178 / 200 | **~89 %** | 9 / 10 |

### Per-dataset results: DRESS-asc vs Betweenness

| Dataset | DRESS wins | Wilcoxon \(p\) | Sig? |
|---------|--------------|----------------|------|
| MUTAG | 20 / 20 | \(< 0.0001\) | \*\*\* |
| PTC_MR | 12 / 20 | 0.18 | - |
| PROTEINS | 18 / 20 | \(< 0.0001\) | \*\*\* |
| DD | 20 / 20 | \(< 0.0001\) | \*\*\* |
| ENZYMES | 16 / 20 | 0.0001 | \*\*\* |
| NCI1 | 17 / 20 | 0.012 | \* |
| IMDB-BINARY | 9 / 20 | 0.98 | - |
| IMDB-MULTI | 9 / 20 | 0.19 | - |
| REDDIT-BINARY | 17 / 20 | 0.0002 | \*\*\* |
| REDDIT-MULTI-5K | 20 / 20 | \(< 0.0001\) | \*\*\* |

DRESS-asc significantly wins on 7 of 10 datasets and never
significantly loses on any.

### Per-dataset results: DRESS-asc vs Fiedler

| Dataset | DRESS wins | Wilcoxon \(p\) | Sig? |
|---------|--------------|----------------|------|
| MUTAG | 14 / 20 | 0.003 | \*\* |
| PTC_MR | 15 / 20 | 0.013 | \* |
| PROTEINS | 16 / 20 | 0.001 | \*\* |
| DD | 18 / 20 | \(< 0.0001\) | \*\*\* |
| ENZYMES | 15 / 20 | 0.017 | \* |
| NCI1 | 16 / 20 | 0.001 | \*\* |
| IMDB-BINARY | 10 / 20 | 0.26 | - |
| IMDB-MULTI | 14 / 20 | 0.003 | \*\* |
| REDDIT-BINARY | 18 / 20 | \(< 0.0001\) | \*\*\* |
| REDDIT-MULTI-5K | 20 / 20 | \(< 0.0001\) | \*\*\* |

DRESS-asc significantly beats Fiedler on 9 of 10 datasets.
Fiedler's spectral cut is competitive on individual biological
datasets but does not generalize across domains.

### Adaptive comparison

BC-adaptive is the theoretical ceiling: it recomputes betweenness
centrality after every single edge removal, at \(O(n \cdot m^2)\) cost.
DRESS-adaptive does the same with DRESS at \(O(k \cdot m^2)\).

| Comparison | Wins | Rate |
|------------|------|------|
| BC-adaptive is most-damaging overall | 167 / 204 | 82 % |
| DRESS-adaptive is most-damaging | 19 / 204 | 9 % |
| DRESS-adaptive vs BC-adaptive | 22 / 200 | 11 % |

BC-adaptive dominates overall.  The gap narrows on dense social
graphs: on IMDB-BINARY and IMDB-MULTI, DRESS-adaptive matches
BC-adaptive (Wilcoxon \(p > 0.25\)).

### Per-dataset mean AUC

| Dataset | DRESS-asc | Betweenness | BC-adaptive | Fiedler | Random |
|---------|-----------|-------------|-------------|---------|--------|
| DD | **0.322** | 0.407 | **0.134** | 0.340 | 0.578 |
| ENZYMES | 0.324 | 0.335 | **0.225** | **0.310** | 0.548 |
| PROTEINS | 0.288 | 0.313 | **0.164** | **0.275** | 0.463 |
| MUTAG | **0.349** | 0.388 | **0.266** | 0.378 | 0.426 |
| PTC_MR | **0.284** | 0.287 | **0.212** | 0.294 | 0.366 |
| NCI1 | **0.299** | 0.306 | **0.200** | 0.317 | 0.360 |
| IMDB-BINARY | 0.453 | **0.448** | **0.434** | 0.477 | 0.809 |
| IMDB-MULTI | 0.497 | **0.495** | **0.478** | 0.532 | 0.827 |
| REDDIT-BINARY | **0.406** | 0.427 | **0.277** | 0.461 | 0.491 |
| REDDIT-MULTI-5K | **0.256** | 0.324 | **0.173** | 0.378 | 0.415 |

Bold values highlight the best static method and the best adaptive
method per dataset.  DRESS-asc is the best static strategy on
6 of 10 datasets.

### Synthetic graphs

| Graph | DRESS-asc | Betweenness | BC-adaptive | DRESS-adaptive | Fiedler | Random |
|-------|-----------|-------------|-------------|---------------|---------|--------|
| BA-200 | 0.702 | 0.750 | **0.445** | 0.615 | **0.628** | 0.746 |
| WS-200 | **0.388** | 0.499 | **0.151** | 0.191 | 0.404 | 0.713 |
| Grid-15×15 | 0.522 | 0.574 | **0.096** | 0.326 | **0.202** | 0.482 |
| StarCliques | **0.120** | **0.120** | 0.128 | **0.109** | 0.118 | 0.389 |

DRESS-asc wins on WS-200, and ties for best static
on StarCliques.  Fiedler dominates on Grid (regular lattice) where
the spectral cut is optimal by design.

## Why it works

DRESS values reflect **local structural support**: an edge \((u,v)\)
receives a high value when \(u\) and \(v\) share many well-connected
common neighbors.  Edges with low DRESS values are structurally exposed -
they sit at bottlenecks, between communities, or at the periphery.

## Cost comparison

| Method | Complexity | Notes |
|--------|------------|-------|
| **DRESS-asc** | \(O(k \cdot m)\) | \(k \approx 7\), static, reuses any DRESS fit |
| Betweenness | \(O(n \cdot m)\) | \(n/k \approx 30\text{–}50\times\) slower |
| Fiedler | \(O(m)\) | Sparse eigensolver; domain-dependent |
| CoreHD | \(O(m)\) | Cheap but weak (rank #7) |
| BC-adaptive | \(O(n \cdot m^2)\) | Theoretical ceiling; impractical at scale |

DRESS-asc achieves 66 % of the adaptive ceiling's improvement
over random, at a cost that is **linear in the number of edges**
and requires no additional computation beyond a standard DRESS fit.
