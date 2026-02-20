# Properties of DRESS

## Boundedness

DRESS values are bounded in \([0, 2]\).

- \(d_{uv} = 2\) if and only if \(u = v\) (self-similarity).
- \(d_{uv} = 0\) cannot occur on any edge in a connected graph (the self-loop
  contribution guarantees a positive numerator).
- For edges between nodes with no common neighbours beyond the self-loop,
  values are small but strictly positive.

## Parameter-free (self-regularisation)

Unlike PageRank (damping factor \(\alpha\)), HITS (normalisation choices), or
GNN-based methods (learning rate, layers, hidden dimensions), DRESS has **no
model parameters**.  The quantities \(\epsilon\) and `max_iterations` are
convergence tolerances, not structural choices.

The nonlinear denominator \(\|u\| \cdot \|v\|\) acts as a natural
regulariser: as edge values grow, the denominator grows proportionally,
automatically bounding the system.  The contraction is built into the formula.

Given a graph, there is exactly one DRESS assignment.  No choices, no
sensitivity analysis.

## Scale invariance (degree-0 homogeneity)

If all edge values are scaled by \(\lambda > 0\), the update rule satisfies:

\[
F(\lambda\, d) = F(d) \quad \forall\, \lambda > 0
\]

This holds because the numerator and denominator have the same degree of
homogeneity in \(d\), so \(\lambda\) cancels.

- numerator is degree 1 in \(d\); denominator is
  \(\sqrt{\cdot} \cdot \sqrt{\cdot}\) = degree 1.  Ratio: degree 0.

!!! note "Design principle"
    The degree-0 condition is **necessary** for boundedness.  Any family member
    where numerator and denominator have mismatched homogeneity will either
    diverge or collapse to zero.

## Unique fixed point

DRESS converges to a unique solution regardless of initialisation.  This
follows from the update being a contraction mapping on the bounded set
\([0, 2]^{|E|}\).

## Interpretation of the fixed point

### Dynamical system interpretation

The DRESS iteration defines a discrete dynamical system on the space of edge
value vectors. 

- **Diffusion.** At each iteration, every edge absorbs similarity information
  from its common-neighbour edges, which in turn absorbed from *their*
  neighbours in the previous step.  After \(t\) iterations, each edge
  effectively integrates structural information from paths of length up
  to \(t\).
- **Self-regulation.** The denominator (product of node norms) acts as an
  automatic gain control: if all values grow, the norms grow proportionally
  and the ratio stays bounded. This is the degree-0 homogeneity property.
- **Contraction.** The combined effect of additive diffusion in the numerator
  and multiplicative normalisation in the denominator produces a contractive
  mapping, guaranteeing convergence to a unique fixed point from any
  non-negative initial condition.

At steady state, every edge value \(d_{uv}\) encodes how structurally similar
nodes \(u\) and \(v\) are, as seen by the entire graph.

- **Normalised structural overlap.**  The numerator aggregates shared
  neighbourhood contributions; the denominator normalises by the structural
  size of each node.  The ratio is a cosine-like similarity between the
  structural profiles of \(u\) and \(v\).
- **Recursive depth.**  Similarity is not based on raw adjacency but on the
  similarity values of neighbouring edges, which in turn depend on *their*
  neighbours, and so on.  At the fixed point every edge has absorbed
  information from the entire connected component.
- **Self-consistency.**  No edge wants to change.  Every value is exactly what
  the DRESS equation predicts from its neighbourhood.  The assignment is the
  unique structural description where every local perspective is globally
  consistent.
- **Diffusive equilibrium.**  Initially all edges are equal (value = 1).
  Information then diffuses through the graph: edges between structurally
  similar nodes reinforce each other, while edges between structurally
  different nodes decay.  The steady state is the equilibrium of this
  diffusion, where values reflect the true structural landscape.

In practice:

- **High \(d_{uv}\):** \(u\) and \(v\) are structurally interchangeable
  (same degree pattern, same community, same role).
- **Low \(d_{uv}\):** \(u\) and \(v\) are structurally different (bridge
  edge, cross-community link, hub-to-leaf).
- **\(d_{uu} = 2\):** a node is maximally similar to itself (the self-loop
  upper bound).

## Determinism

Given the same graph, DRESS always produces the same values.  There is no
randomness, no sampling, no ordering dependence.

## Isomorphism invariance

If two graphs \(G\) and \(G'\) are isomorphic, their sorted DRESS edge value
multisets are identical.  DRESS depends only on the graph's structure (adjacency
and weights), not on vertex labelling.  Any relabelling (permutation of nodes)
that maps \(G\) to \(G'\) also maps each edge's neighbourhood identically,
so the fixed-point equation produces the same values.

This makes the sorted DRESS vector a **graph fingerprint**: two graphs are
isomorphism candidates if and only if their fingerprints match.  See
[Graph Isomorphism](../applications/isomorphism.md) for details.

## Numerical stability

Many iterative graph algorithms suffer from numerical issues in
floating-point arithmetic: vanishing or exploding values, sensitivity to
initialisation, or accumulation of rounding errors across iterations.
DRESS is numerically stable by construction.  Four properties work together
to guarantee this:

1. **Bounded output.**  All values remain in \([0, 2]\).  There is no risk of
   overflow, and the self-loop contribution prevents underflow to zero.

2. **Degree-0 homogeneity.**  Scaling all values by a constant \(\lambda\)
   cancels out in the next iterate: \(F(\lambda\,d) = F(d)\).  This means
   small multiplicative perturbations (the dominant form of floating-point
   error) do not amplify across iterations.

3. **Self-regularising denominator.**  The \(\|u\| \cdot \|v\|\) term grows
   proportionally to the numerator.  If rounding errors push values slightly
   upward, the denominator absorbs the increase automatically.  There is no
   separate normalisation step that could introduce additional error.

4. **Contraction mapping.**  Each iteration strictly reduces the distance
   between distinct value vectors.  Rounding errors introduced at iteration
   \(t\) are contracted in iteration \(t+1\), rather than accumulated.

Together these properties mean DRESS does not require extended precision,
careful ordering of operations, or compensated summation to produce reliable
results.  The same double-precision implementation yields bit-identical
results across platforms (verified by the cross-validation test suite).

## Low computational complexity

DRESS runs in \(O(N + k \cdot E \cdot \bar{d})\) time and \(O(N + E)\)
memory, where \(k\) is the number of iterations (typically 5--20) and
\(\bar{d}\) is the average degree.  For sparse graphs (\(E = O(N)\)), the
total cost is effectively **linear in the graph size**.  With precomputed
neighbourhood intercepts the per-edge cost drops further to
\(O(|N[u] \cap N[v]|)\).  See
[Complexity](equation.md#complexity) for the full analysis.

## Massive parallelism

Each edge update reads only from its neighbourhood's current values.  All edges
can be updated simultaneously (Jacobi-style iteration), making DRESS trivially
parallelisable with OpenMP, CUDA, or any SIMD framework.

## Local invertibility (incremental edge query)

DRESS is a global recursive fixed point: changing one edge should, in
principle, require re-solving the entire system.  Remarkably, each edge's
value can be computed from its **immediate neighbourhood alone**, without
re-fitting.

For any edge \((u, v)\) (existing, removed, or hypothetical) its DRESS
value satisfies a scalar fixed-point equation:

\[
\hat{d}_{uv} = \frac{A + c\,d_{uv}}
  {\sqrt{D_u + c\,d_{uv}} \;\cdot\; \sqrt{D_v + c\,d_{uv}}}
\]

where:

- \(A = \displaystyle\sum_{x \in N[u] \cap N[v]}
  \bigl(\bar{w}(u,x)\,d_{ux} + \bar{w}(v,x)\,d_{vx}\bigr)\) is the
  common-neighbour contribution (known from the fitted graph),
- \(D_u = \displaystyle\sum_{x \in N[u]} \bar{w}(u,x)\,d_{ux}\) and
  \(D_v = \displaystyle\sum_{x \in N[v]} \bar{w}(v,x)\,d_{vx}\) are the
  squared node norms without the queried edge,
- \(c = \bar{w}(u,v)\) accounts for the edge appearing in both the
  numerator and denominator through the self-loop path.

This is a scalar equation in \(d_{uv}\) alone and converges in a few
iterations (or can be solved in closed form).

**Why this matters.**  DRESS is global yet **locally invertible**: any single
edge value is fully determined by its immediate neighbourhood at steady state.
This means incremental updates are cheap (\(O(\deg u + \deg v)\) per query
after one \(O(E)\) fit), and the global fixed point barely moves when a
single edge is perturbed.

**Empirical validation (leave-one-out).** Each edge is removed in turn, its
value is predicted from the remaining graph, and the prediction is compared to
the original fitted value:

| Graph | \(\|V\|\) | \(\|E\|\) | Pearson \(r\) | \(R^2\) |
|-------|-----------|-----------|---------------|--------|
| Facebook | 4 040 | 88 234 | 0.9997 | 0.9990 |
| Amazon | 548 552 | 925 872 | 0.9989 | 0.9899 |
| YouTube | 1 157 828 | 2 987 624 | 0.9997 | 0.9977 |
| Wiki-Vote | 7 115 | 103 689 | 0.9998 | 0.9980 |
| Zachary Karate Club | 35 | 78 | 0.9993 | 0.9877 |

The high \(R^2\) confirms that the local closed-form equation is an accurate
proxy for the global fixed point, even on small graphs where single-edge
removal is a proportionally larger perturbation.

## Comparison with related methods

Several existing algorithms compute node- or edge-level scores via iterative
processes.  DRESS differs from all of them in important ways.

### SimRank

[SimRank](https://dl.acm.org/doi/10.1145/775047.775126) (Jeh & Widom, 2002)
measures **node–node** similarity: two nodes are similar if their neighbours
are similar.

| Aspect | SimRank | DRESS |
|--------|---------|-------|
| Entity | Node pairs | Edges |
| Damping factor | Required (\(C \in (0,1)\)); results change with \(C\) | **None** (parameter-free) |
| Bounded | \([0, 1]\) | \([0, 2]\) |
| Complexity (naïve) | \(O(k \cdot N^2 \cdot \bar{d}^2)\) | \(O(k \cdot E \cdot \bar{d})\) |
| Memory | \(O(N^2)\) (all pairs) | \(O(N + E)\) (edges only) |
| Unique fixed point | Depends on \(C\) | **Always** |
| Self-similarity | \(\text{sim}(u,u) = 1\) by definition | \(d_{uu} = 2\) (derived from formula) |

SimRank's \(O(N^2)\) memory makes it impractical on large graphs.  Optimised
variants reduce cost but introduce additional parameters or approximations.

### PageRank

[PageRank](https://dl.acm.org/doi/10.1016/S0169-7023(98)00110-X) (Brin & Page, 1998)
computes a **node importance** score via a random-walk model.

| Aspect | PageRank | DRESS |
|--------|----------|-------|
| Entity | Nodes | Edges |
| Damping factor | Required (\(\alpha\), typically 0.85); results change with \(\alpha\) | **None** |
| Linearity | Linear eigenvector problem | Nonlinear fixed point |
| Output | Global importance ranking | Per-edge structural similarity |
| Complexity | \(O(k \cdot E)\) | \(O(k \cdot E \cdot \bar{d})\) |
| Parameters | \(\alpha\), personalisation vector (optional) | **Zero** |

PageRank answers a different question (*which nodes are important?*) and
cannot distinguish structural roles of edges.  DRESS and PageRank are
complementary: PageRank ranks nodes, DRESS characterises edges.

### Weisfeiler–Leman (WL) colour refinement

The 1-WL algorithm iteratively refines node colours based on neighbour
multisets.  It is the basis of most GNN message-passing architectures.

DRESS can be understood as a **continuous relaxation of 1-WL**.
Both algorithms iterate over the same local structure — each node's
neighbourhood — and converge to a fixed point.  The key difference is
that 1-WL produces **discrete colour partitions** while DRESS produces
**continuous real-valued edge scores**.

| Aspect | 1-WL | DRESS |
|--------|------|-------|
| Entity | Nodes (discrete colours) | Edges (continuous values) |
| Output | Colour histogram (partition) | Real-valued edge vector (metric) |
| Refinement | Hash of neighbour multiset | Cosine-like ratio with recursive weights |
| Fixed point | Stable colouring (finite steps) | Unique continuous fixed point |
| Sensitivity | Cannot distinguish regular graphs | Same theoretical limits (CFI, SRG) |
| Parameters | None | None |

This continuous relaxation has concrete advantages:

- **Metric output.**  1-WL answers "same or different"; DRESS answers
  "how similar."  Every binary classification task becomes a
  regression/similarity task, and every histogram becomes a distribution.
- **Edge granularity.**  1-WL assigns one colour per node; DRESS assigns
  one value per edge, providing strictly finer structural description.
- **Downstream utility.**  Continuous values can be thresholded, ranked,
  clustered, or used directly as features — none of which are possible
  with a discrete partition.

### Summary

| Property | SimRank | PageRank | 1-WL | **DRESS** |
|----------|---------|----------|------|-----------|
| Parameter-free | ✗ | ✗ | ✓ | **✓** |
| Edge-level output | ✗ | ✗ | ✗ | **✓** |
| Bounded | ✓ | ✓ | — | **✓** |
| Unique fixed point | ✗ | ✓ | ✓ | **✓** |
| Memory | \(O(N^2)\) | \(O(N)\) | \(O(N)\) | **\(O(N+E)\)** |
| Handles weighted/directed | Partial | ✓ | ✗ | **✓** |
