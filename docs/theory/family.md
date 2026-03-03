# The DRESS Family of Equations

DRESS is not a single equation. It is a **family** of continuous structural
refinement algorithms, building from the concrete to the general. The paper
presents them in this order: Original-DRESS → Motif-DRESS → Generalized-DRESS
→ ∇-DRESS / Δ-DRESS.

## Motif-DRESS

Original-DRESS is limited to triangle neighborhoods. Motif-DRESS generalizes it by defining $\mathcal{N}_M(u,v)$ as a symmetric vertex neighborhood: the set of endpoints of edges in instances of a motif $M$ containing edge $(u,v)$ that are adjacent to $u$ or $v$, always including $u$ and $v$ themselves, with $f = \text{sum}$, $g = \|u\| \cdot \|v\|$ (product of geometric means), and an optional symmetric weight function $\bar{w}: E \to \mathbb{R}_{>0}$ (with $\bar{w}_e = 1$ for unweighted graphs):

\[
d_{uv}^{(t+1)} = \frac{\displaystyle\sum_{x \in \mathcal{N}_M(u,v)} \bigl(\bar{w}_{ux}\,d_{ux}^{(t)} + \bar{w}_{xv}\,d_{xv}^{(t)}\bigr)}{\|u\|^{(t)} \cdot \|v\|^{(t)}}
\]

where $\|u\|^{(t)} = \sqrt{\sum_{x \in N[u]} \bar{w}_{ux} \cdot d_{ux}^{(t)}}$ is the vertex norm, and $\bar{w}_{ab} \cdot d_{ab} = 0$ whenever $(a,b) \notin E$.

The weights $\bar{w}(e)$ act as a multiplicative factor, controlling how much structural information flows along each edge. Because the weights appear identically in the numerator and denominator (both are degree-1 in $\bar{w} \cdot d$), uniformly scaling all weights does not change the fixed point; only the relative weights matter.

Different choices of motif $M$ yield different neighborhood operators:

- **Triangle ($M = K_3$):** $\mathcal{N}_{K_3}(u,v) = N[u] \cap N[v]$, the closed common neighborhood. This recovers Original-DRESS exactly.
- **4-cycle ($M = C_4$):** For each 4-cycle $u$-$x$-$y$-$v$, $\mathcal{N}_{C_4}(u,v)$ contains $u$, $v$, $x$, and $y$.
- **$K_4$ clique:** For each pair $x, y$ with $\{u,v,x,y\}$ forming a $K_4$, $\mathcal{N}_{K_4}(u,v)$ contains $u$, $v$, $x$, and $y$.

### Properties

**Complexity:** $\mathcal{O}(\text{Motif Extraction}) + \mathcal{O}\!\bigl(I \cdot \sum_{e} |\mathcal{N}_M(e)|\bigr)$, where $I$ is the number of iterations and $\sum_e |\mathcal{N}_M(e)|$ is the total size of all motif-neighborhood lists. For Original-DRESS (triangles), this reduces to $\mathcal{O}(I \cdot m \cdot \Delta)$. For motifs such as 4-cycles or $K_4$ cliques in sparse graphs, the motif-neighborhood lists can be significantly smaller, making Motif-DRESS faster than Original-DRESS.

**Invariant: $d_{uu} = 2$.** The self-similarity is invariant under iteration: since $\mathcal{N}_M(u,u)$ includes all edges incident to $u$ symmetrically, and $d_{ux} = d_{xu}$, substituting $v = u$ yields $d_{uu}^{(t+1)} = 2\|u\|^2 / \|u\|^2 = 2$ for all $t$. Thus $d_{uu} = 2$ is a fixed property of the equation, not a free parameter.

### Expressiveness

All experiments below use the $K_4$ clique motif. The specific SRG pairs tested below are known to be indistinguishable by 3-WL; each successful distinction therefore demonstrates that Motif-DRESS empirically exceeds 3-WL on these instances.

- **Rook vs. Shrikhande:** Successfully distinguishes this pair of SRGs with parameters $(16, 6, 2, 2)$. The Rook graph ($K_4 \square K_4$) contains $K_4$ cliques while the Shrikhande graph does not, so the $K_4$-neighborhood sizes differ per edge.
- **Chang Graphs:** Distinguishes 3 of the 6 pairwise comparisons among the four SRGs with parameters $(28, 12, 6, 4)$: T(8) vs each of Chang-1, Chang-2, and Chang-3. The three Chang graphs are pairwise indistinguishable by Motif-$K_4$ (all three have identical $K_4$-neighborhood structure per edge).

### Proof Sketch of Convergence

For Motif-DRESS to converge to a unique fixed point, the aggregation function $f$ and vertex norm $g$ must satisfy the following three conditions:

1. **Scale Invariance (Degree-0 Homogeneity).** If $f$ is positively homogeneous of degree $p$ and $g$ is positively homogeneous of degree $q$, then the ratio is homogeneous of degree $p - 2q$. For the iteration to remain bounded and non-trivial: $\boxed{p = 2q}$. If $p > 2q$ the iteration diverges; if $p < 2q$ it collapses to zero. This makes the mapping scale-invariant: multiplying all $d$-values by a constant leaves the next iterate unchanged, so the result depends only on graph structure.

2. **Boundedness.** The motif neighborhood $\mathcal{N}_M(u,v)$ is always a subset of each vertex's full neighborhood, so the numerator decomposes as $\sum_{\mathcal{N}_M} \bar{w}_{ux}\,d_{ux} + \sum_{\mathcal{N}_M} \bar{w}_{vx}\,d_{vx} \leq \|u\|^2 + \|v\|^2$. Self-loop inclusion ensures $\|u\|,\,\|v\| > 0$, giving the finite per-step bound:

    $$F(d)_{uv} \;\leq\; \frac{\|u\|}{\|v\|} + \frac{\|v\|}{\|u\|}$$

    By AM-GM this is always $\geq 2$, with equality when $\|u\| = \|v\|$.  In the **unweighted** case (uniform $\bar{w}$), the fixed-point contraction forces the norm ratio to remain bounded, and the converged values satisfy $d^* \in [0, 2]$.  With **non-uniform edge weights**, vertex norms can differ even for structurally identical vertices, so fixed-point values may exceed 2.

3. **Contraction on the Hilbert Projective Metric.** Since $F$ is a positive, degree-0 homogeneous map on the cone $\mathbb{R}_{>0}^{|E|}$, Birkhoff's contraction theorem guarantees that $F$ is a strict contraction under the Hilbert projective metric $d_H(x, y) = \log\!\bigl(\max_{e} x_e/y_e \cdot \max_{e} y_e/x_e\bigr)$, provided $F$ maps a bounded part of the cone into a strictly smaller part — which follows from the finite per-step bound above. By the Banach fixed-point theorem on the complete metric space $(\mathbb{R}_{>0}^{|E|}/\!\sim,\, d_H)$, the iteration converges to a unique ray, and the boundedness step pins the representative to a finite vector $d^*$ (in the unweighted case, $d^* \in [0, 2]^{|E|}$). A complete formal verification of the contraction constant is deferred to future work; all empirical tests confirm convergence within 20 iterations.

**Self-loops.** Self-loops are added to every vertex before iteration (i.e., the algorithm uses the closed neighborhood $N[u] = N(u) \cup \{u\}$). The self-loop edge $(u,u)$ participates in both the aggregation and the vertex norm; without it, an isolated edge with no common neighbors would produce $g(u,v) = 0$, making the iteration undefined.

### Original-DRESS as a Special Case

The Original-DRESS equation, introduced in [Castrillo, León & Gómez (2018)](https://arxiv.org/abs/1805.01419), is a special case of Motif-DRESS with:

- **Motif:** Triangle ($M = K_3$), so $\mathcal{N}_{K_3}(u,v) = N[u] \cap N[v]$
- **Aggregation:** $f = \text{sum}$
- **Norm:** $g(u,v) = \|u\| \cdot \|v\|$ (geometric norm), where $\|u\| = \sqrt{\sum_{x \in N[u]} d_{ux}^{(t)}}$

\[
d_{uv}^{(t+1)} = \frac{\sum_{x \in N[u] \cap N[v]} \bigl(d_{ux}^{(t)} + d_{xv}^{(t)}\bigr)}{\|u\|^{(t)} \cdot \|v\|^{(t)}}
\]

where $N[u] = N(u) \cup \{u\}$ denotes the closed neighborhood of $u$, and $\|u\|^{(t)} = \sqrt{\sum_{x \in N[u]} d_{ux}^{(t)}}$ is the vertex norm. Note that $d_{uu} = 2$ is invariant under iteration. This equation converges to a unique fixed point, providing a continuous structural fingerprint for the graph. However, because it aggregates strictly over triangles (common neighbors), it cannot distinguish graphs with identical triangle counts per edge (e.g., Strongly Regular Graphs).

### Some Valid Family Members

Any aggregation–norm pair satisfying $p = 2q$ produces a valid DRESS variant.
Below are concrete examples that can be combined with any motif neighborhood operator.

#### Minkowski-r-DRESS

\[
\phi_r = \sum_x (w_{ux}\,d_{ux})^r + (w_{vx}\,d_{vx})^r
\qquad
\|u\| = \Bigl(\sum_{x \in N[u]} (w_{ux}\,d_{ux})^r\Bigr)^{1/2}
\]

so that $\phi$ has degree $r$ and each norm factor has degree $r/2$,
giving $p = r$ and $2q = 2 \cdot r/2 = r$. ✓

| $r$ | Aggregation behavior | Note |
|-------|----------------------|------|
| 1 | **SUM** (linear) | The Original-DRESS equation |
| 2 | Squared contributions dominate | More weight on strong edges |
| $\to\infty$ | Max over common neighbors | Only the strongest edge pair matters |

#### Cosine-DRESS

\[
\phi = \sum_x w_{ux}\,d_{ux} \cdot w_{vx}\,d_{vx}
\qquad
\|u\| = \sqrt{\sum_{x \in N[u]} (w_{ux}\,d_{ux})^2}
\]

$\phi$ is degree 2 and each norm is degree 1, so $p = 2q = 2$. ✓

This variant measures similarity through **correlated** edge strengths rather
than their sum.

#### Geometric-DRESS

\[
\phi = \sum_x \sqrt{w_{ux}\,d_{ux} \cdot w_{vx}\,d_{vx}}
\qquad
\|u\| = \Bigl(\sum_{x \in N[u]} w_{ux}\,d_{ux}\Bigr)^{1/2}
\]

$\phi$ is degree 1 and each norm is degree $1/2$, so $p = 2q = 1$. ✓

Uses the geometric mean of edge pairs. May be more robust to outlier edges.

#### Kernel-DRESS

Replace the aggregation with a positive-definite kernel:

\[
\phi_K = \sum_x K(w_{ux}\,d_{ux},\; w_{vx}\,d_{vx})
\]

This is valid provided $K$ is positively homogeneous of degree $2q$ for
the chosen norm. For example, with a polynomial kernel
$K(a,b) = (a + b)^r$ paired with the appropriate Minkowski norm.

#### k-hop-DRESS

Given graph $G$, construct the $k$-hop augmented graph $G_k$ (where
$(u,v) \in G_k$ iff a path of length $\le k$ exists in $G$), then apply
any DRESS family member to $G_k$. No formula changes are needed; the
higher-order structure comes entirely from the input graph construction.

All proofs (boundedness, convergence, uniqueness) carry over immediately since
DRESS is well-defined on any valid weighted graph.  Note that with non-uniform
edge weights, vertex norms may differ and converged values can exceed 2
(see [Proof Sketch – Boundedness](#proof-sketch-of-convergence)).

## Generalized-DRESS

Motif-DRESS fixes the aggregation to summation and the norm to the product of geometric means. Generalized-DRESS is the most abstract template, allowing any choice of these components as long as the resulting update rule preserves the convergence guarantees (degree-0 homogeneity, boundedness, and contraction). For each edge $(u,v)$:

\[
d^{(t+1)} = \frac{f\bigl(\mathbf{d}^{(t)},\, \mathcal{N},\, \bar{w}\bigr)}{g\bigl(\mathbf{d}^{(t)},\, \mathcal{N},\, \bar{w}\bigr)}
\]

where $d \equiv d_{uv}$ is the similarity value assigned to edge $(u,v)$, and:

- $\mathcal{N}(u,v)$ is a **symmetric neighborhood operator** — the structural context aggregated for $(u,v)$,
- $\bar{w}: E \to \mathbb{R}_{>0}$ is a **symmetric weight function** ($\bar{w}(u,v) = \bar{w}(v,u)$; $\bar{w} \equiv 1$ for unweighted graphs),
- $f$ is the **aggregation function**,
- $g$ is the **norm function**.

Because $\mathcal{N}$ and $\bar{w}$ are symmetric, $f$ and $g$ receive the same inputs for $(u,v)$ and $(v,u)$, so $d(u,v) = d(v,u)$ holds for every member of the family. For Original-DRESS and Motif-DRESS this follows directly from the equation; in the general case it is guaranteed by the symmetry of the inputs.

Original-DRESS and Motif-DRESS are both special cases: Original-DRESS fixes $\mathcal{N}$ to triangles, $f = \text{sum}$, $g = \|u\| \cdot \|v\|$ (product of geometric means), $\bar{w} \equiv 1$; Motif-DRESS generalizes $\mathcal{N}$ to arbitrary motifs and $\bar{w}$ to non-uniform weights while keeping the same $f$ and $g$. Generalized-DRESS opens all four parameters, enabling variants such as Cosine-DRESS (cosine similarity aggregation) or Minkowski-$r$ norms.

## Δ-DRESS

Δ-DRESS breaks symmetry by running standard DRESS on each vertex-deleted subgraph $G \setminus \{v\}$ for every $v \in V$. The graph fingerprint is the sorted multiset of $n$ converged edge-value vectors (one per deletion), compared by flattening and sorting without any summarization.

Unlike approaches that modify the DRESS iteration itself (e.g., clamping edge values), Δ-DRESS runs *unmodified* DRESS on structurally altered graphs. Deleting a vertex from a regular graph produces an irregular subgraph where standard DRESS can now distinguish structure that was hidden by the uniform regularity.

See [Δ^k-DRESS (Iterated Deletion)](delta-ell-dress.md) for the generalization to depth $k$.

## Family Structure

The DRESS variants form a nested hierarchy with one orthogonal composition operator:

$$\text{Generalized-DRESS} \supset \text{Motif-DRESS} \supset \text{Original-DRESS}$$

$\Delta(\cdot)$ is an **orthogonal wrapper** applicable to any of the above: given any DRESS variant $\mathcal{F}$,

$$\Delta^k\text{-DRESS}(\mathcal{F}) = h\!\left(\bigsqcup_{|S|=k} \mathcal{F}(G \setminus S)\right)$$

The deletion strategy is independent of the choice of $\mathcal{N}$, $f$, $g$, and $\bar{w}$. All experiments in the higher-order sections use $\mathcal{F} = \text{Original-DRESS}$, but $\Delta$-Motif-DRESS or $\Delta$-Cosine-DRESS are equally valid and may offer complementary expressiveness. See [$\Delta^k$-DRESS](delta-ell-dress.md) for details.

## k-Ego-DRESS

For each vertex $v$, extract its $k$-hop ego network (the induced subgraph on all vertices within distance $k$ from $v$) and run Original-DRESS on that subgraph independently. The graph fingerprint is the multiset of these $n$ local fixed points.

- **Complexity:** $n \times \mathcal{O}(\text{DRESS}(N_k(v)))$. Extremely fast on sparse graphs where ego networks are small.
- **Expressiveness:** Captures local structural variation that global DRESS averages out. On vertex-transitive graphs (e.g., Petersen), 2-hop Ego DRESS splits nodes into distinct structural classes based on their local neighborhood topology.


