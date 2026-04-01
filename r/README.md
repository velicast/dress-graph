# dress.graph (R)

**A Continuous Framework for Structural Graph Refinement**

DRESS is a deterministic, parameter-free framework that iteratively refines the structural similarity of edges in a graph to produce a canonical fingerprint: a real-valued edge vector, obtained by converging a non-linear dynamical system to its unique fixed point. The fingerprint is self-contained, isomorphism-invariant by construction, guaranteed bitwise-equal across any vertex labeling, numerically stable (no overflow, no error amplification, no undefined behavior), fast and embarrassingly parallel to compute: DRESS total runtime is O(I * m * d_max) for I iterations to convergence, and convergence is guaranteed by Birkhoff contraction.

## Install

CRAN hosts a stable release. For the latest version, install from GitHub.

```r
# From CRAN (stable)
install.packages("dress.graph")

# From GitHub (latest)
# install.packages("remotes")
remotes::install_github("velicast/dress-graph", subdir="r")
```

## Quick start

```r
library(dress.graph)

result <- fit(
  n_vertices = 4L,
  sources    = c(0L, 1L, 2L, 0L),
  targets    = c(1L, 2L, 3L, 3L)
)
print(result$edge_dress)
```

For the full API and documentation, see the [main repository](https://github.com/velicast/dress-graph).
