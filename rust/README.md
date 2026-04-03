# dress-graph (Rust)

**A Continuous Framework for Structural Graph Refinement**

DRESS is a deterministic, parameter-free framework for continuous structural graph refinement. It iterates a nonlinear dynamical system on real-valued edge similarities and produces a graph fingerprint as a sorted edge-value vector once the iteration reaches a prescribed stopping criterion. The resulting fingerprint is self-contained, isomorphism-invariant by construction, reproducible across vertex labelings under the reference implementation, numerically robust in practice, and efficient to compute with straightforward parallelization and distribution.

## Quick start

```rust
use fit::{DRESS, Variant};

let sources = vec![0, 1, 2, 0];
let targets = vec![1, 2, 3, 3];

let mut g = DRESS::new(4, sources, targets,
                       None, None, Variant::Undirected, false).unwrap();
g.fit(100, 1e-6);
let result = g.result();
println!("{:?}", result.edge_dress);
```

For the full API and documentation, see the [main repository](https://github.com/velicast/dress-graph).
