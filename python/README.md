# dress-graph (Python)

**A Continuous Framework for Structural Graph Refinement**

DRESS is a deterministic, parameter-free framework for continuous structural graph refinement. It iterates a nonlinear dynamical system on real-valued edge similarities and produces a graph fingerprint as a sorted edge-value vector once the iteration reaches a prescribed stopping criterion. The resulting fingerprint is self-contained, isomorphism-invariant by construction, reproducible across vertex labelings under the reference implementation, numerically robust in practice, and efficient to compute with straightforward parallelization and distribution.

## Install

```bash
pip install dress-graph
```

## Quick start

```python
from dress import fit

result = fit(
    n_vertices=4,
    sources=[0, 1, 2, 0],
    targets=[1, 2, 3, 3],
)
print(result.edge_dress)  # DRESS value for each edge
```

For the full API and documentation, see the [main repository](https://github.com/velicast/dress-graph).
