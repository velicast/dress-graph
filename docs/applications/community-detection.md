# Community Detection

## Approach

DRESS edge values form a natural spectrum: intra-community edges (supported by
many common neighbours) receive high values, while inter-community edges
(bridges with few common neighbours) receive low values.

Thresholding edges by DRESS value separates communities **without training
data** and with minimal hyperparameters (only the threshold itself, or none at
all if combined by local community definitions).

## SCAN improvement

The SCAN (Structural Clustering Algorithm for Networks) algorithm uses a
structural similarity threshold \(\varepsilon\) to identify core nodes.
Replacing SCAN's raw similarity metric with DRESS values:

- **Reduces sensitivity** to the \(\varepsilon\) parameter.
- **Improves NMI** (Normalised Mutual Information) on LFR benchmarks.
- **Improves Modularity** on LFR benchmarks.
- **Community size distributions** closer to the ground truth (Less singletons).

This improvement was introduced in
[arXiv:1805.01419](https://arxiv.org/abs/1805.01419).

## Agglomerative hierarchical algorithm

Beyond improving existing algorithms, a dedicated **agglomerative hierarchical
clustering algorithm** was designed around DRESS and benchmarked in the
ASOMAN 2017 and arXiv:1805.12238 papers.  The algorithm:

1. Computes DRESS values for all edges.
2. Uses the DRESS values as the merging criterion in an agglomerative
   (bottom-up) hierarchical process, producing a multi-scale dendrogram.
3. Extends naturally to **overlapping communities**: the disjoint structure is
   first obtained, then a membership probability function based on DRESS
   values assigns nodes to multiple communities, yielding both fuzzy and
   crisp overlapping partitions.

The algorithm was evaluated on standard LFR benchmark graphs and compared with
state-of-the-art methods, achieving high-quality disjoint and overlapping
community structure on large-scale complex networks.

## References

- E. Castrillo, E. León, J. Gómez. *Dynamic Structural Similarity on Graphs.*
  arXiv:1805.01419.
  [arXiv](https://arxiv.org/abs/1805.01419)
- E. Castrillo, E. León, J. Gómez. *Fast Heuristic Algorithm for Multi-Scale
  Hierarchical Community Detection.* ASONAM 2017.
  [ACM](https://dl.acm.org/citation.cfm?doid=3110025.3110125)
- E. Castrillo, E. León, J. Gómez. *High-Quality Disjoint and Overlapping
  Community Structure in Large-Scale Complex Networks.* arXiv:1805.12238.
  [arXiv](https://arxiv.org/abs/1805.12238)
