# Research Directions

DRESS is a continuous relaxation of 1-WL colour refinement: same
discriminative power, but with **real-valued edge scores** instead of
discrete colours.  This means DRESS can, in principle, replace 1-WL in
every application that relies on it.  Below are open research directions
where the continuous output may provide concrete advantages.

---

## 1. Graph Neural Networks

GNNs perform differentiable WL:
\(h_v \leftarrow \sigma(W \cdot \text{AGG}(\{h_u : u \in N(v)\}))\).

DRESS values can serve as **pre-converged structural edge features**:

```python
edge_attr = dress_forward_backward(graph)   # shape [E, 2]
out = gnn(x=node_features, edge_index=edges, edge_attr=edge_attr)
```

Because DRESS encodes the full 1-WL information in two floats per edge,
the network starts with a structurally meaningful initialisation rather
than learning these features from scratch.  This is analogous to
random-walk or Laplacian positional encodings used in Graphormer and GPS,
but deterministic and structurally grounded.

---

## 2. Graph Kernels

The WL subtree kernel builds a colour histogram at each refinement round
and sums inner products across rounds.  DRESS replaces this with a single
continuous edge-value vector that auto-converges (no round count
hyperparameter \(h\)).

A DRESS kernel can be defined via Earth Mover's Distance:

\[
k(G, H) = \exp\!\bigl(-\gamma \cdot \text{EMD}(\text{dress}(G),\,
\text{dress}(H))\bigr)
\]

Wasserstein-based comparisons are smoother than histogram intersection and
naturally handle graphs of different sizes.

**Potential applications:** molecular property prediction, protein function
classification.

---

## 3. Symmetry Breaking in Combinatorial Optimisation

1-WL gives an equitable partition: nodes in the same colour class are
structurally equivalent (same orbit).  DRESS gives the same partition,
plus a **continuous ranking** within each class via edge values.

```
1. Compute DRESS on the constraint graph.
2. Nodes with identical DRESS profiles are in the same orbit.
3. Fix the node with smallest DRESS norm in each orbit.
4. Prune symmetric branches.
```

The continuous values provide a natural tie-breaking order for branching
heuristics in integer linear programming solvers.

---

## 4. Database Query Optimisation

Colour refinement on query graphs produces structural summaries for
estimating join sizes.  DRESS assigns each edge a continuous "structural
importance" score that can feed a regression model for cardinality
estimation, rather than discrete histogram lookups.  DRESS also handles
weighted property graphs natively.

---

## 5. Canonical Labelling

Tools like nauty use colour refinement to produce an initial partition,
then branch and backtrack.  Running DRESS first yields a finer initial
ordering:

```
1. Compute F-DRESS + B-DRESS: each edge gets a (f, b) pair.
2. Node signature = sorted list of incident (f, b) pairs.
3. Identical signatures = same orbit (guaranteed).
4. Feed this partition to nauty as initial colouring.
```

If the continuous values break ties that early WL rounds leave unresolved,
nauty's search tree shrinks.

---

## 6. Graph Compression and Summarisation

The 1-WL quotient graph has one super-node per colour class.  DRESS
produces the same quotient structure, but each super-edge carries a
**continuous weight** (the DRESS value of that edge type).  This tells you
not just *that* two groups are connected, but *how tightly*.

**Potential applications:** knowledge-graph compression, network
visualisation.

---

## 7. Role Extraction in Networks

1-WL assigns one colour per structural role (binary: same or different).
DRESS gives a **continuous role embedding** per node:

```python
role(u) = sorted([dress(e) for e in edges_incident_to(u)])
```

Two nodes with identical embeddings are structurally equivalent (same as
1-WL).  But you can also compute role *similarity*:

\[
\text{role\_sim}(u, v) = 1 - \frac{\|r(u) - r(v)\|}{\|r(u)\| + \|r(v)\|}
\]

In social network analysis, a CEO and a VP are not identical roles, but
they are more similar to each other than to an intern.  DRESS gives this
gradient naturally; 1-WL only gives a binary answer.

---

## 8. Molecular Fingerprints

Morgan/ECFP fingerprints are WL on molecular graphs with atom labels:
hash each atom's neighbourhood into a bit vector for similarity search.

A DRESS molecular fingerprint:

```python
def dress_fingerprint(mol):
    d = DRESS(mol.num_atoms, mol.bond_sources, mol.bond_targets)
    result = d.fit()
    return sorted(result.edge_dress)
```

Advantages over ECFP:

- **No radius hyperparameter.** DRESS auto-converges.
- **Continuous.** Supports Wasserstein distance, not just Tanimoto on bit
  vectors.
- **No hash collisions.** ECFP uses 32/64-bit hashes; DRESS values are
  continuous and collision-free up to float precision.
- **Metric.** Distance between DRESS fingerprints is a meaningful
  structural dissimilarity.

**Potential applications:** virtual screening, QSAR modelling, chemical
space visualisation.

---

## 9. Approximate Homomorphism Counting

Two graphs are 1-WL equivalent if and only if they agree on
\(\text{hom}(T, \cdot)\) for all trees \(T\).  Since DRESS has the same
discriminative power, the same equivalence holds.

DRESS adds a quantitative layer: the distance between two DRESS
signatures correlates with the difference in tree-homomorphism counts.
This could serve as a fast proxy for the full tree-homomorphism profile
without expensive exact computation.

---

## Summary

| Application | 1-WL gives | DRESS adds |
|---|---|---|
| Graph isomorphism | yes/no | **how different** (metric) |
| GNNs | theoretical ceiling | **structural edge features** (pre-converged) |
| Graph kernels | histogram kernel | **Wasserstein kernel** (no \(h\) hyperparameter) |
| Symmetry breaking | orbit partition | **continuous tie-breaking** |
| Query optimisation | discrete bins | **regression features** |
| Canonical labelling | initial partition | **finer initial ordering** |
| Compression | quotient graph | **weighted quotient** (cohesion measure) |
| Role extraction | binary same/different | **continuous role similarity** |
| Molecular fingerprints | bit vector (ECFP) | **continuous, collision-free, metric** |
| Homomorphism counting | exact equivalence | **approximate distance** |

The recurring theme: DRESS replaces 1-WL's discrete output with a
continuous, metric one.  Every binary classification becomes a
regression/similarity task, and every histogram becomes a distribution.
