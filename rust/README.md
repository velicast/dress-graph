# dress-graph (Rust)

**A Continuous Framework for Structural Graph Refinement**

DRESS is a provably continuous relaxation of the Weisfeiler–Leman algorithm.
At depth k, higher-order DRESS is **provably at least as powerful as (k+2)-WL**
in expressiveness — the base algorithm (k=0) already matches 2-WL, and each
level adds one WL dimension.
Yet it is dramatically cheaper to compute: a single DRESS run costs
O(I · m · d_max) where I is the number of iterations, and depth-k requires
C(n,k) independent runs — a total of O(C(n,k) · I · m · d_max), compared to
O(n^(k+3)) for (k+2)-WL.  Space complexity is O(n + m), compared to
O(n^(k+2)) for (k+2)-WL.
The algorithm is embarrassingly parallel in two orthogonal ways — across the
C(n,k) subproblems and across edge updates within each iteration — enabling
distributed/cloud and multi-core/GPU/SIMD implementations.

## Quick start

```rust
use dress_graph::{DressGraph, Variant};

let result = DressGraph::new(4, &[0,1,2,0], &[1,2,3,3], None, Variant::Undirected, false)
    .fit(100, 1e-6);
println!("{:?}", result.edge_dress);
```

For the full API and documentation, see the [main repository](https://github.com/velicast/dress-graph).
