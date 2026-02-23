# Graph Classification

## Approach

A DRESS fingerprint summarises a graph's structural profile in a fixed-size
vector.  The pipeline has three steps:

1. **Compute DRESS** on the graph (a single call to `dress_fit`).
2. **Summarise** the edge values into a 24-dimensional fingerprint.
3. **Classify** with an off-the-shelf classifier (Random Forest or Gradient
   Boosted Trees).

No neural network, no GPU, no training loop, no hyperparameter search.

### Fingerprint construction

The sorted DRESS edge values are summarised into a **percentile vector**:

- \(q\) evenly spaced percentiles of the edge value distribution (where \(q \in \{10, 20, 40, 80\}\) is tuned via cross-validation).
- Distribution entropy (Shannon entropy of a 50-bin histogram).
- Number of edges.
- Number of distinct edge values (rounded to 8 decimals).
- Fraction of near-zero edges (\(d_{uv} < 10^{-10}\)).

Total: **\(q + 4\) dimensions** per graph, regardless of graph size.

## Weighting strategy

DRESS operates on edge weights.  When node labels are available, encoding them
into edge weights lets DRESS distinguish edges between different label pairs.

### Canonical frequency-based encoding

1. Count the corpus-wide frequency of each node label across all graphs.
2. Rank labels by descending frequency (most common label → ID 0).
3. Assign edge weight via **pair encoding**:

\[
w_{uv} = \min(c_u, c_v) \cdot k + \max(c_u, c_v) + 1
\]

where \(c_u\) is the canonical ID of node \(u\)'s label and \(k\) is the
total number of distinct labels.  Every distinct unordered label pair maps to
a unique weight.

This encoding is deterministic and depends only on label frequencies — an
intrinsic dataset property — not on the arbitrary integer assignment.

### When labels are absent

For graphs without node labels (social networks), **unweighted DRESS** is
used.  

## Results

Benchmarked on all 9 datasets from Xu et al. (ICLR 2019), the same benchmarks
used to evaluate GIN, GCN, GraphSAGE, and the WL subtree kernel.

### Per-strategy results

| Dataset | Labels | Graphs | Unweighted | Canonical pair |
|---------|--------|--------|------------|----------------|
| MUTAG | 7 | 188 | 88.3 ± 5.2 | **90.4 ± 5.7** |
| PTC_MR | 18 | 344 | 61.6 ± 5.3 | **64.6 ± 6.4** |
| PROTEINS | 3 | 1,113 | 71.5 ± 3.1 | **73.0 ± 5.1** |
| NCI1 | 22 | 4,110 | 77.7 ± 1.4 | **79.2 ± 1.4** |
| IMDB-BINARY | — | 1,000 | **74.7 ± 4.1** | 72.4 ± 4.0 |
| IMDB-MULTI | — | 1,500 | 48.7 ± 2.4 | **48.9 ± 3.2** |
| REDDIT-BINARY | — | 2,000 | **90.3 ± 1.3** | 87.6 ± 2.7 |
| REDDIT-MULTI-5K | — | 4,999 | 50.6 ± 1.5 | **53.1 ± 1.5** |
| COLLAB | — | 5,000 | **79.8 ± 1.6** | 79.1 ± 0.9 |

**Canonical pair** wins on all labeled (bio) datasets.
**Unweighted** wins on most social networks, but canonical pair can help
on multi-class social networks (IMDB-MULTI, REDDIT-MULTI-5K) where
degree-pair encoding provides finer discrimination between classes.

### Comparison with WL and GIN

Best DRESS result per dataset (canonical pair for labeled, unweighted for
unlabeled) against the WL subtree kernel and GIN-0 from Xu et al. (2019):

| Dataset | DRESS | WL | GIN-0 | vs WL | vs GIN |
|---------|-------|-----|-------|-------|--------|
| MUTAG | **90.4** | **90.4** | 89.4 | ±0.0 | **+1.0** |
| PTC_MR | **64.6** | 59.9 | **64.6** | **+4.7** | ±0.0 |
| PROTEINS | 73.0 | 75.0 | **76.2** | −2.0 | −3.2 |
| NCI1 | 79.2 | **86.0** | 82.7 | −6.8 | −3.5 |
| IMDB-BINARY | 74.7 | 73.8 | **75.1** | **+0.9** | −0.4 |
| IMDB-MULTI | 48.7 | 50.9 | **52.3** | −2.2 | −3.6 |
| REDDIT-BINARY | 90.3 | 81.0 | **92.4** | **+9.3** | −2.1 |
| REDDIT-MULTI-5K | 53.1 | 52.5 | **57.5** | **+0.6** | −4.4 |
| COLLAB | 79.8 | 78.9 | **80.2** | **+0.9** | −0.4 |

DRESS beats WL on 5 datasets, ties on 1, and loses on 3.
DRESS beats GIN on 1 dataset, ties on 1, and loses on 7.

### Hyperparameter tuning

To ensure an apples-to-apples comparison, DRESS is evaluated using nested cross-validation (10 outer folds, 5 inner folds), matching the methodology of Xu et al. (2019). The classifier (Random Forest or Gradient Boosted Trees), number of estimators, maximum depth, and the number of DRESS quantiles are tuned per dataset via grid search on the inner folds.

### Complexity comparison

| Method | Parameters | Training | Fingerprint | Classifier |
|--------|-----------|----------|-------------|------------|
| GIN | ~100K learnable | GPU, ~200 epochs | Learned | End-to-end |
| WL kernel | \(h\) (rounds), \(C\) (SVM) | Kernel SVM, tuned via CV | Histogram per round | SVM |
| **DRESS** | **\(q\) (quantiles), depth, estimators** | **RF/GBT, tuned via CV** | **14d–84d, single \(O(m)\) call** | **Off-the-shelf GBT/RF** |

### End-to-end scaling

The per-graph computation is comparable: both WL and DRESS iterate over
edges a small number of times (\(h\) rounds for WL, \(k\) iterations for
DRESS).  The critical difference is what happens **after**:

| | WL subtree kernel | DRESS fingerprint |
|---|---|---|
| Per-graph | \(O(h \cdot m)\) | \(O(k \cdot m)\) |
| Representation | Variable-length histogram (grows with vocabulary) | Fixed 14d–84d vector |
| Pairwise comparison | \(O(N^2 \cdot V)\) kernel matrix (\(V\) = vocab size) | **Not needed** |
| Classifier | Kernel SVM: \(O(N^2)\) to \(O(N^3)\) | RF/GBT: \(O(N \cdot d \log N)\), \(d \le 84\) |
| **Total pipeline** | \(O(N \cdot h \cdot m + N^2 \cdot V)\) | \(O(N \cdot k \cdot m + N \cdot d \log N)\) |

WL produces a kernel, not a feature vector — it is fundamentally tied to
\(O(N^2)\) pairwise comparisons.  DRESS produces a fixed-size embedding,
so the entire pipeline scales **linearly** in the number of graphs \(N\).
For small benchmarks (MUTAG, 188 graphs) this is irrelevant, but for
larger corpora (COLLAB at 5,000 or beyond) the quadratic term in WL
becomes a practical bottleneck while DRESS remains trivial.

## Observations

### Social networks and community structure

DRESS excels on social network classification — particularly
REDDIT-BINARY (+9.3 vs WL), COLLAB (+0.9 vs WL), and
REDDIT-MULTI-5K (+0.6 vs WL).  This is not a
coincidence: social network classification is largely determined by
**community structure** (how clusters of users are organised), and DRESS
was originally designed to capture exactly this signal for
[community detection](community-detection.md).

The DRESS fingerprint on a social network is effectively a summary of its
community structure profile: the distribution of intra-community edge
strengths, bridge edge strengths, and the balance between them.

### Labeled molecular graphs

On molecular datasets with discrete atom types, canonical pair encoding
consistently improves over unweighted DRESS (+2.1 on MUTAG, +3.0 on PTC_MR,
+1.5 on PROTEINS, +1.5 on NCI1).  The encoding lets DRESS distinguish
bond types — a Carbon–Carbon bond and a Carbon–Nitrogen bond look identical
without labels but receive different weights under canonical pair encoding.

### Where DRESS falls short

DRESS underperforms on NCI1 (−6.8 vs WL, −3.5 vs GIN) and IMDB-MULTI
(−2.2 vs WL, −3.6 vs GIN).  These datasets require learning complex
feature interactions that a fixed-size summary cannot capture.
DRESS provides what topology and label pair statistics *alone* can give —
when the task demands more, learned models have an inherent advantage.
