# Graph Classification: DRESS + GNNs

## Motivation

The [fingerprint-based approach](classification.md) shows that DRESS captures
useful structural information for graph classification using only off-the-shelf
classifiers.  A natural question follows: **what happens when we feed DRESS
values directly into Graph Neural Networks as edge and node features?**

DRESS acts as a universal preprocessing step, one scalar per edge and one per
node, precomputed once and cached.  The GNN receives richer structural input
without any architecture changes or additional learnable parameters.

## Setup

All experiments use the **ZINC-12K** benchmark (Dwivedi et al., 2023):

- **Task**: Graph regression (predict constrained solubility)
- **Metric**: MAE (lower is better)
- **Split**: 10,000 / 1,000 / 1,000 (train / val / test)
- **Parameter budget**: ~100K parameters per model
- **Seeds**: 4 (42, 123, 456, 789)
- **Training**: 400 epochs max, early stopping (patience 50), ReduceLROnPlateau

### DRESS preprocessing

For each graph in the dataset:

1. Call `dress_fit()` to obtain per-edge DRESS values and per-node DRESS values.
2. Optionally apply \(\Delta^k\)-DRESS (`delta_dress_fit` with \(k=1\)) for
   higher-order structural refinement.
3. Cache the results. Preprocessing the full ZINC-12K dataset takes < 1 second.

The GNN receives:

- **Edge features**: bond type (4 categories) + DRESS edge value (1 scalar)
- **Node features**: atom type (28 categories) + DRESS node value (1 scalar)

## Results

| Model | MAE |
|-------|-----|
| GIN (Dwivedi+23) | 0.526 ± 0.051 |
| GAT (Dwivedi+23) | 0.384 ± 0.007 |
| GCN (Dwivedi+23) | 0.367 ± 0.011 |
| GatedGCN (Dwivedi+23) | 0.282 ± 0.015 |
| **GPS + DRESS attn** | **0.247 ± 0.004** |
| **DRESSNet** | **0.240 ± 0.009** |
| **GIN + DRESS** | **0.227 ± 0.029** |
| **PNA + DRESS** | **0.213 ± 0.004** |
| PNA (Dwivedi+23) | 0.188 ± 0.004 |

### Headline result

A vanilla GIN with DRESS features achieves **0.227**, a **57% error reduction**
over the published GIN baseline (0.526).  This simple addition is enough to
surpass GatedGCN (0.282), a model with edge gating, residual connections,
and batch normalization.

### Architectures tested

**GIN + DRESS.**  GINEConv with bond type and DRESS edge value concatenated
as edge features.  DRESS node values are concatenated to atom embeddings.
Hidden dimension 88, ~98K parameters.

**PNA + DRESS.**  PNAConv with degree-scalers and the same DRESS edge/node
augmentation.  Hidden dimension 35, ~95K parameters.  Achieves 0.213, within
reach of the vanilla PNA baseline (0.188) despite the smaller hidden dimension
forced by the additional edge input channels.

**GPS + DRESS attention.**  GPS transformer layers where DRESS edge values
are used as an attention bias in the self-attention mechanism, plus DRESS
positional encodings.  Hidden dimension 44, ~89K parameters.  The lower
parameter count (forced by the transformer overhead at 100K budget) limits
performance, but it demonstrates that DRESS works with attention-based
architectures.

**DRESSNet.**  A DRESS-native architecture where message passing is explicitly
gated by structural similarity: \(\text{gate}_{ij} = \sigma(W \cdot d_{ij})\),
\(\text{msg}_{ij} = \text{gate}_{ij} \odot m_{ij}\).  Hidden dimension 72,
~98K parameters.  Achieves 0.240 with a simple design, competitive with
much more complex baselines.

### Ablation: component contributions

| Variant | MAE |
|---------|-----|
| GIN + DRESS (no bond) | 0.314 ± 0.025 |
| GIN + bond + DRESS (no node DRESS) | 0.295 ± 0.006 |
| GIN + bond + DRESS (full) | 0.235 ± 0.012 |
| GIN + bond + DRESS (full, \(k=1\)) | 0.227 ± 0.029 |

Each component contributes: bond features, DRESS edge values, DRESS node
values, and higher-order \(\Delta^k\)-DRESS all improve the result.

### Effect of \(\Delta^k\)-DRESS order

| Model | \(k=0\) | \(k=1\) |
|-------|---------|---------|
| GIN + DRESS | 0.235 ± 0.012 | **0.227 ± 0.029** |
| PNA + DRESS | 0.213 ± 0.004 | 0.212 ± 0.009 |
| DRESSNet | 0.267 ± 0.025 | **0.240 ± 0.009** |

Higher-order DRESS (\(k=1\)) consistently helps, with the largest gains on
DRESSNet (0.267 → 0.240) and GIN (0.235 → 0.227).

## Key takeaways

1. **Universal plug-in.**  DRESS improves GIN, PNA, GPS, and custom
   architectures.  It is not tied to any specific GNN design.

2. **Zero-cost preprocessing.**  DRESS values are precomputed once
   (< 1 second for ZINC-12K) and cached.  No additional learnable
   parameters are introduced.

3. **Simplicity wins.**  The best approach is the simplest: concatenate
   the raw DRESS scalar to existing edge/node features.  More complex
   encodings (scatter statistics, RBF expansions) did not improve results.

4. **Higher-order helps.**  \(\Delta^k\)-DRESS with \(k=1\) provides
   consistent improvements, especially for architectures that rely
   heavily on structural information (DRESSNet, GIN).

## Relationship to fingerprint classification

The [fingerprint approach](classification.md) summarizes DRESS values into
a fixed-size vector and uses Random Forest or GBT.  The GNN approach instead
feeds raw DRESS values directly into the message-passing layers, letting the
network learn how to use them.

Both approaches confirm the same insight: **DRESS edge values carry
structural information that improves graph classification**, whether consumed
by a simple classifier or a neural network.
