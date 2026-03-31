# DRESS.jl (Julia)

**A Continuous Framework for Structural Graph Refinement**

DRESS is a deterministic, parameter-free framework that iteratively refines the structural similarity of edges in a graph to produce a canonical fingerprint: a real-valued edge vector, obtained by converging a non-linear dynamical system to its unique fixed point. The fingerprint is self-contained, isomorphism-invariant by construction, guaranteed bitwise-equal across any vertex labeling, numerically stable (no overflow, no error amplification, no undefined behavior), fast and embarrassingly parallel to compute: DRESS total runtime is O(I * m * d_max) for I iterations to convergence, and convergence is guaranteed by Birkhoff contraction.

## Quick start

```julia
using DRESS

result = dress_fit(4, [0, 1, 2, 0], [1, 2, 3, 3])
println(result.edge_dress)
```

For the full API and documentation, see the [main repository](https://github.com/velicast/dress-graph).
