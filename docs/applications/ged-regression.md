# Graph Edit Distance (GED) Regression

## Task

Given two graphs \(G_1\) and \(G_2\), predict their **Graph Edit Distance (GED)** - the minimum number of node and edge insertions, deletions, and substitutions required to transform \(G_1\) into \(G_2\).

Exact GED computation is NP-hard and practically impossible for graphs with more than a dozen nodes. Most modern approaches use heavy Graph Neural Networks (GNNs) to learn a heuristic approximation.

DRESS offers a lightweight, training-free alternative: we extract the DRESS fingerprint for each graph and train a simple regression model (like Gradient Boosted Trees) on the difference between the fingerprints.

## Method

For a pair of graphs \((G_1, G_2)\), we compute a feature vector representing their structural difference:

1. **Absolute Fingerprint Difference**: \(| \text{fp}_1 - \text{fp}_2 |\), where \(\text{fp}\) is the 84-dimensional DRESS percentile fingerprint.
2. **Size Differences**: \(|n_1 - n_2|\) and \(|m_1 - m_2|\).
3. **Wasserstein Distance**: The Earth Mover's Distance between the raw, unbinned DRESS edge value distributions of the two graphs.

This combined feature vector is fed into a standard Gradient Boosting Regressor to predict the continuous GED value.

## Datasets

We evaluate on three standard GED benchmark datasets (Bai et al., 2019):

- **LINUX**: 800 train / 200 test program call graphs (unlabeled).
- **AIDS700nef**: 560 train / 140 test molecular graphs (29 unique node labels).
- **IMDBMulti**: 1200 train / 300 test dense social graphs (unlabeled).

For labeled graphs (AIDS700nef), we apply **canonical pair weighting**: node labels are ranked by corpus frequency and each edge is assigned a weight encoding the ordered label pair of its endpoints. This makes DRESS label-aware without any learned parameters.

## Results

We compare the DRESS-based regressor against published neural baselines like SimGNN (Bai et al., 2019) and TaGSim (Bai & Zhao, 2022).

| Dataset | Method | MSE (\(\times 10^{-3}\)) | Spearman \(\rho\) | Kendall \(\tau\) | \(p@20\) |
|---------|--------|--------------------------|-------------------|------------------|----------|
| **LINUX** | SimGNN | 12.840 | 0.884 | 0.753 | 0.193 |
| | TaGSim | 5.278 | 0.941 | 0.834 | **0.867** |
| | **DRESS** | **0.343** | **0.988** | **0.935** | 0.635 |
| **AIDS700nef** | SimGNN | 14.769 | 0.534 | 0.397 | 0.024 |
| | TaGSim | 5.890 | 0.679 | 0.520 | 0.266 |
| | **DRESS** | **4.681** | **0.730** | **0.574** | **0.304** |
| **IMDBMulti** | SimGNN | 77.871 | 0.715 | 0.626 | 0.886 |
| | TaGSim | **35.690** | **0.958** | **0.926** | **0.986** |
| | **DRESS** | 157.739 | 0.424 | 0.363 | 0.028 |

### Analysis

- **LINUX**: DRESS massively outperforms published baselines on MSE, \(\rho\), and \(\tau\), achieving MSE \(0.343 \times 10^{-3}\) - **15× lower than TaGSim** - with near-perfect Spearman correlation (0.988). TaGSim retains an edge on the precision-at-top metric \(p@20\).
- **AIDS700nef**: DRESS achieves state-of-the-art across all four metrics, reducing MSE by 21% relative to TaGSim and improving Spearman \(\rho\) from 0.679 to 0.730.
- **IMDBMulti**: Performance drops significantly on dense social graphs. This is a known theoretical limitation: DRESS matches 2-WL in expressiveness, but still struggles to distinguish highly symmetric, dense structures (like cliques and stars) common in social networks.
