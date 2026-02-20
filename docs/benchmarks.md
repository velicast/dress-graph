# Benchmarks

## Convergence on Real-World Graphs

The table below reports DRESS convergence on several well-known graph datasets.
All runs used convergence tolerance \(\epsilon = 10^{-6}\) with a maximum of
100 iterations.

| Graph | Vertices | Edges | Iterations | Final delta |
|-------|----------|-------|------------|-------------|
| Amazon product co-purchasing | 548,552 | 925,872 | 18 | 6.35e-7 |
| Wiki-Vote | 8,298 | 103,689 | 17 | 8.31e-7 |
| LiveJournal social network | 4,033,138 | 27,933,062 | 30 | 7.09e-7 |
| Facebook (konect) | 59,216,215 | 92,522,012 | 26 | 6.84e-7 |
| Facebook (UCI/UNI) | 58,790,783 | 92,208,195 | 26 | 6.84e-7 |

### Key Observations

- **Low iteration count.** Even on graphs with tens of millions of vertices and
  edges, DRESS converges in fewer than 31 iterations — consistent with the
  [contraction-mapping guarantee](theory/properties.md#unique-fixed-point).
- **Scale independence.** Iteration count grows very slowly with graph size.
  A graph with 59 M vertices needs only ~1.5× the iterations of one with 8 K
  vertices.
- **Uniform residual.** The final \(\delta\) is consistently on the order of
  10⁻⁷, indicating that convergence quality does not degrade with graph size.
