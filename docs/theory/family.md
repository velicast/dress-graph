# The DRESS Family of Equations

DRESS is not a single equation. It is a **family** parameterised by the choice
of aggregation function and norm.  Any member of the family that satisfies the
conditions below inherits all of the properties proved for the SUM variant
(boundedness, unique fixed point, convergence, parameter-free operation).

## General template

For each edge \((u, v)\):

\[
d_{uv} = \frac{\phi\bigl(\{(w_{ux}\,d_{ux},\; w_{vx}\,d_{vx})\}_{x \in N[u] \cap N[v]}\bigr)}
  {\|u\|_\alpha \cdot \|v\|_\alpha}
\]

where:

- \(\phi\) is the **aggregation function** over common-neighbour edge pairs,
- \(\|\cdot\|_\alpha\) is the **node norm**.

## Necessary condition: degree-0 homogeneity

If the aggregation \(\phi\) is positively homogeneous of degree \(p\) and the
norm is positively homogeneous of degree \(q\), then the ratio is homogeneous of
degree \(p - 2q\).  For the fixed-point iteration to remain bounded and
non-trivial:

\[
\boxed{p = 2q}
\]

- \(p > 2q\): the iteration **diverges**.
- \(p < 2q\): the iteration **collapses to zero**.

Degree-0 homogeneity is exactly the condition that makes the mapping
**scale-invariant**: multiplying all \(d\)-values by a constant leaves the
next iterate unchanged, which is why the result depends only on the graph
structure and not on the initial values.

## Valid family members

Any aggregation–norm pair satisfying \(p = 2q\) produces a valid DRESS variant.
Below are concrete examples.

### Minkowski-\(r\) DRESS

\[
\phi_r = \sum_{x}\, (w_{ux}\,d_{ux})^r + (w_{vx}\,d_{vx})^r
\qquad
\|u\|_r = \Bigl(\sum_{x \in N[u]} (w_{ux}\,d_{ux})^r\Bigr)^{1/r}
\]

Here \(\phi_r\) is homogeneous of degree \(r\) and \(\|u\|_r\) is homogeneous
of degree 1, so the ratio is homogeneous of degree \(r - 2 \cdot \frac{r}{2}\)
... let us verify: the norm raised to degree \(r\) gives \(\sum (w d)^r\), so
the norm itself is degree-1 in \(d\), and the denominator is degree 2.
\(\phi_r\) is degree \(r\) in \(d\), so we need \(r = 2\) for degree-0.

More precisely, the correctly paired version is:

\[
\phi_r = \sum_x (w_{ux}\,d_{ux})^r + (w_{vx}\,d_{vx})^r
\qquad
\|u\| = \Bigl(\sum_{x \in N[u]} (w_{ux}\,d_{ux})^r\Bigr)^{1/2}
\]

so that \(\phi\) has degree \(r\) and each norm factor has degree \(r/2\),
giving \(p = r\) and \(2q = 2 \cdot r/2 = r\). ✓

| \(r\) | Aggregation behaviour | Note |
|-------|----------------------|------|
| 1 | **SUM** (linear) | The standard DRESS equation |
| 2 | Squared contributions dominate | More weight on strong edges |
| \(\to\infty\) | Max over common neighbours | Only the strongest edge pair matters |

### Product (COSINE) DRESS

\[
\phi = \sum_x w_{ux}\,d_{ux} \cdot w_{vx}\,d_{vx}
\qquad
\|u\| = \sqrt{\sum_{x \in N[u]} (w_{ux}\,d_{ux})^2}
\]

\(\phi\) is degree 2 and each norm is degree 1, so \(p = 2q = 2\). ✓

This variant measures similarity through **correlated** edge strengths rather
than their sum.

### Geometric DRESS

\[
\phi = \sum_x \sqrt{w_{ux}\,d_{ux} \cdot w_{vx}\,d_{vx}}
\qquad
\|u\| = \Bigl(\sum_{x \in N[u]} w_{ux}\,d_{ux}\Bigr)^{1/2}
\]

\(\phi\) is degree 1 and each norm is degree \(1/2\), so \(p = 2q = 1\). ✓

Uses the geometric mean of edge pairs.  May be more robust to outlier edges.

### Kernel DRESS

Replace the aggregation with a positive-definite kernel:

\[
\phi_K = \sum_x K(w_{ux}\,d_{ux},\; w_{vx}\,d_{vx})
\]

This is valid provided \(K\) is positively homogeneous of degree \(2q\) for
the chosen norm.  For example, with a polynomial kernel
\(K(a,b) = (a + b)^r\) paired with the appropriate Minkowski norm.

## Higher-order DRESS

Given graph \(G\), construct the \(k\)-hop augmented graph \(G_k\) (where
\((u,v) \in G_k\) iff a path of length \(\le k\) exists in \(G\)), then apply
any DRESS family member to \(G_k\).  No formula changes are needed; the
higher-order structure comes entirely from the input graph construction.

All proofs (boundedness, convergence, uniqueness) carry over immediately since
DRESS is well-defined on any valid weighted graph.
