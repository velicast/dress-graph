# Applications

DRESS is a **single formula** that simultaneously enables solutions to problems
that normally require completely different algorithms:

- [**Graph Isomorphism**](isomorphism.md): fingerprinting graphs for fast comparison
- [**Community Detection**](community-detection.md): unsupervised edge classification into intra- and inter-community

## Relevant domains

| Domain | Use case |
|--------|----------|
| Social networks | Link prediction, community detection, friend recommendation |
| Fraud detection | Anomalous edges in transaction graphs (low DRESS = structurally unusual) |
| Knowledge graphs | Predicting missing relations, entity resolution |
| Network security | Detecting structural changes in communication graphs |
| Drug discovery | Molecular graph fingerprinting (atoms = nodes, bonds = edges) |
| Compiler optimisation | Dependency graph deduplication via isomorphism detection |

## Potential applications

The following are **promising research directions** where DRESS's properties
suggest it could be useful.  These areas are still being explored;
prototyping and benchmarking are ongoing.

!!! info "Work in progress"
    Benchmark tools for each application are currently being built under the
    `tools/` directory of the repository.  These will provide reproducible
    evaluation pipelines so that DRESS can be compared head-to-head against
    existing methods on standard datasets.

### Anomaly detection in dynamic graphs

Since DRESS converges to a unique fixed point deterministically, comparing
DRESS values across graph snapshots could localise structural changes:

1. Fit \(G\) at time \(t_1\), fit \(G'\) at time \(t_2\).
2. Edges with large \(|d_{uv}^{t_1} - d_{uv}^{t_2}|\) may indicate where
   structural change occurred.

Early LOO reconstruction results suggest that the baseline is stable
(MAE < 1 % on large graphs), which would mean deviations above that baseline
are likely real structural changes rather than noise.  Further validation is
needed.

### Edge role discovery

Edges with similar DRESS values may occupy similar structural positions.
Clustering edges by value could yield structural roles (bridge edges,
clique-internal edges, peripheral edges) without defining roles a priori.
This hypothesis remains to be validated on diverse graph families.

### Graph fingerprinting

Sorting the DRESS edge values produces a canonical descriptor of a graph.
The \(L^2\) distance between sorted DRESS vectors could define a useful
metric over graphs.  Preliminary isomorphism results are encouraging
(100 % on MiVIA / IsoBench), but the discriminative power of this
fingerprint as a general graph metric is still under study.

For directed graphs, the fingerprint is constructed by running **F-DRESS**
(FORWARD) and **B-DRESS** (BACKWARD) separately, then representing each edge
as a pair \((f_{uv}, b_{uv})\).  Sorting these pairs lexicographically breaks
transpose-symmetric ambiguities that a single D-DRESS pass cannot resolve
(e.g. star graphs where \(G\) and \(G^\top\) yield identical D-DRESS vectors).
See [Graph Isomorphism](isomorphism.md#directed-graphs) for details.

### Network robustness

Edges with low DRESS values appear to be structurally isolated (few common
neighbours, weak support), potential bridges and bottlenecks.  If this
holds broadly, removing low-DRESS edges first could provide a principled
attack / failure ordering.  This needs systematic benchmarking against
established robustness measures.

### Node-level applications via \(D_u\)

The DRESS fixed point naturally produces a **node-level quantity**:

\[
D_u \;=\; \sum_{x \in N[u]} \bar{w}(u,x)\,d_{ux}
\]

This is the weighted sum of DRESS values around node \(u\) (equivalently, the
squared node norm that appears in the denominator of the equation).  It is
computed as a by-product of fitting. No extra work is needed.

\(D_u\) could be useful wherever a **structural node score** is needed:

- **Node centrality.**  High \(D_u\) means the node's edges are
  well-supported by common neighbours.  Unlike degree centrality, \(D_u\)
  accounts for the quality of connections, not just their count.  Unlike
  betweenness or closeness, it is computed in the same \(O(t \cdot |E|)\)
  pass as the edge values.
- **Node anomaly detection.**  A node with low \(D_u\) relative to its
  degree, i.e.\ low \(D_u / \deg(u)\), has many connections that lack
  structural support.  In a social network this could flag spam accounts;
  in a transaction graph, fraudulent actors.
- **Community seed selection.**  Nodes with the highest \(D_u\) sit deep
  inside tightly-knit regions.  They could serve as natural seeds for
  community expansion or label propagation.
- **Influence maximisation.**  If high-\(D_u\) nodes tend to be embedded
  in cohesive clusters, they may also be effective seed nodes for
  information-diffusion cascades, though this needs empirical validation.
- **GNN node features.**  \(D_u\) (and the normalised form
  \(D_u / \deg(u)\)) can be concatenated onto node feature vectors before
  message passing, giving the GNN access to structural context at no
  additional training cost.

These applications are **preliminary hypotheses**; systematic benchmarks
against established centrality and anomaly measures are still needed.

### Topological data analysis

DRESS values could define a filtration on edges: sweeping a threshold from
2 down to 0, adding edges in decreasing order.  The resulting persistent
homology might capture intrinsic topological features of unweighted graphs.
This connection has not yet been formally investigated.

## Integration with GNNs and LLMs

DRESS is not a replacement for Graph Neural Networks or Large Language Models.
It is a **complementary structural primitive** that can strengthen both.

### DRESS as edge features for GNNs

Standard GNN architectures (GCN, GAT, GraphSAGE) propagate *node* features
through message passing.  Edge-level structural information is typically
limited to binary adjacency or hand-crafted features.  DRESS values provide a
**parameter-free, pre-computed edge feature** that encodes global structural
context:

- **Input enrichment.** Concatenate or add \(d_{uv}\) to the edge feature
  vector before message passing.  This gives the GNN access to higher-order
  structural information without increasing the number of layers.
- **Attention guidance.** In attention-based GNNs (GAT), DRESS values can
  serve as an attention prior or bias, directing the model to weight
  structurally significant edges more heavily.
- **Edge classification.** For tasks like link prediction or relation
  classification, DRESS values provide a strong unsupervised baseline that a
  GNN can refine with task-specific supervision.

Because DRESS is deterministic and parameter-free, it adds **zero learnable
parameters** and introduces no training instability.

### DRESS for graph-aware LLMs

As LLMs are increasingly applied to graph-structured data (knowledge graphs,
code dependency graphs, molecule graphs), a key challenge is **how to inject
structural information into the token stream**.  DRESS offers a natural
solution:

- **Graph tokenisation.** When serialising a graph into a prompt, annotate
  each edge with its DRESS value.  This gives the LLM a compact structural
  signal that distinguishes bridge edges from clique-internal edges, context
  that is lost in a flat adjacency list.
- **Retrieval-augmented generation (RAG).** In graph-based RAG pipelines,
  DRESS values can rank edges by structural importance, guiding the retriever
  to fetch the most informative subgraph neighbourhoods.
- **Graph fingerprints as embeddings.** The sorted DRESS vector is a
  fixed-size, deterministic graph descriptor.  It can serve as a graph-level
  embedding for retrieval or comparison tasks without requiring a learned
  encoder.
