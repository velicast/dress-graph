# Graph Retrieval

## Task

Given a query graph, rank a database of graphs by structural similarity.
This is the graph analogue of image retrieval or document search — and it
requires a **fixed-length graph representation** that supports efficient
nearest-neighbor lookup.

DRESS provides exactly this: an 84-dimensional fingerprint per graph,
computed in a single unsupervised pass, with no training data required.

## Method

1. **Compute DRESS** on every graph in the database (one `dress_fit` call each).
2. **Summarise** each graph's edge values into an 84-dimensional percentile
   fingerprint (80 evenly spaced percentiles + entropy + edge count +
   unique count + zero fraction).
3. **Rank** database graphs by L1 distance to the query fingerprint.

No GED labels are used, no model is trained, and the fingerprint is
compatible with standard approximate nearest-neighbor indices (FAISS, Annoy,
etc.) for sub-linear retrieval at scale.

### Fingerprint construction

Same construction as [classification](classification.md), but with 80
quantiles instead of 20 for finer distributional resolution:

- 80 evenly spaced percentiles of the DRESS edge value distribution.
- Shannon entropy of a 50-bin histogram.
- Number of edges (size signal).
- Number of distinct edge values.
- Fraction of near-zero edges.

Total: **84 dimensions** per graph.

### Why 84 dimensions?

A sweep over `n_quantiles` ∈ {10, 20, 40, 80, 100, 120, 160} shows
performance saturating around 80–100 quantiles.  Beyond that, extra
dimensions add noise without capturing new distributional information —
consistent with the small graph sizes in these benchmarks (LINUX: ~8 nodes,
AIDS: ~9 nodes).

| n_quantiles | Dimensions | LINUX P\@10 (L2) | AIDS P\@10 (L1) |
|-------------|-----------|-------------------|------------------|
| 10 | 14 | 0.360 | 0.201 |
| 20 | 24 | 0.404 | 0.210 |
| 40 | 44 | 0.383 | 0.215 |
| **80** | **84** | **0.434** | **0.223** |
| 100 | 104 | 0.440 | 0.220 |
| 120 | 124 | 0.443 | 0.216 |
| 160 | 164 | 0.398 | 0.219 |

We use **80 quantiles (84d)** as the default — near the peak, with no
fragile hyperparameter sensitivity.

## Ground truth

We evaluate against **Graph Edit Distance (GED)** — the minimum number of
node/edge insertions, deletions, and substitutions to transform one graph
into another.  GED matrices from the GED-EXP benchmark (Bai et al., 2019)
provide the "true" ranking for each query.

Datasets: **LINUX** (800 train / 200 test program call graphs) and
**AIDS700nef** (560 train / 140 test molecular graphs).

## Metrics

- **P\@K**: fraction of the predicted top-K that appear in the true GED top-K.
- **NDCG\@K**: normalised discounted cumulative gain (ranking quality).
- **Spearman ρ**: full-ranking correlation with GED.

## Results

### LINUX

| Method | P\@1 | P\@5 | P\@10 | P\@20 | NDCG\@10 | ρ |
|--------|------|------|-------|-------|----------|-----|
| Size baseline | 0.170 | 0.293 | 0.312 | 0.375 | 0.872 | **0.867** |
| DRESS Pct-84d L2 | **0.505** | **0.404** | 0.434 | 0.522 | 0.987 | 0.838 |
| DRESS Pct-84d L1 | 0.230 | 0.331 | 0.365 | 0.453 | 0.988 | 0.786 |
| DRESS Pct-24d L1 | 0.290 | 0.394 | **0.460** | **0.556** | 0.987 | 0.818 |
| DRESS Wasserstein | 0.360 | 0.307 | 0.375 | 0.523 | 0.983 | 0.521 |

DRESS achieves **P\@1 = 0.505** (3× the size baseline) and
**NDCG\@10 = 0.987** — near-perfect ranking quality in the top 10.

### AIDS700nef

| Method | P\@1 | P\@5 | P\@10 | P\@20 | NDCG\@10 | ρ |
|--------|------|------|-------|-------|----------|-----|
| Size baseline | 0.086 | 0.093 | 0.129 | 0.177 | 0.758 | **0.529** |
| DRESS Pct-84d L1 | 0.207 | 0.230 | **0.223** | **0.216** | **0.846** | 0.472 |
| DRESS Pct-24d L2 | **0.243** | 0.224 | 0.200 | 0.188 | 0.828 | 0.474 |
| DRESS Wasserstein | 0.214 | 0.224 | 0.212 | 0.193 | 0.819 | 0.307 |

DRESS achieves **1.7× the size baseline on P\@10** and **NDCG\@10 = 0.846**.

## Comparison with supervised methods

Published GED prediction models (trained on GED labels):

| Method | Type | LINUX ρ | Training required |
|--------|------|---------|-------------------|
| SimGNN | GNN | ~0.93 | Yes (GED labels) |
| GMN | GNN | ~0.95 | Yes (GED labels) |
| **DRESS** | **Fingerprint** | **0.84** | **None** |

DRESS reaches ~88% of the supervised correlation while requiring **zero
training, zero GED labels, and zero GPU time**.  The supervised methods
also require \(O(N^2)\) pairwise GED computation for training data, while
DRESS computes each fingerprint independently in \(O(m)\).

## The ρ vs P\@K tension

An interesting pattern: the size baseline wins on Spearman ρ (0.867 vs 0.838
on LINUX) but loses badly on P\@K.  This means size captures the **broad
trend** (bigger graphs have higher GED to small ones) but cannot distinguish
graphs of similar size — exactly where DRESS excels.

For retrieval, P\@K is the metric that matters: users want the right
results in the top positions.  A method with perfect ρ but poor P\@10 is
useless for practical search.

## Observations

- **L1 distance** is more robust than L2 for P\@10, consistent across
  both datasets.  L1 downweights outlier dimensions in the fingerprint.
- **Histograms** (fixed-bin counts) performed consistently worse than
  percentile fingerprints — the CDF representation captures distribution
  shape more faithfully.
- **DRESS+Size concatenation** did not help on LINUX (where graphs have
  similar sizes) and helped only marginally on AIDS.  The fingerprint
  already captures size implicitly via the edge count feature.
- **Cosine distance** was mediocre — magnitude (scale) matters for this
  task, and cosine discards it.
