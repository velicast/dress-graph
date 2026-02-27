#ifndef NABLA_DRESS_H
#define NABLA_DRESS_H

#include "dress/dress.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compute the Nablavidualized^k-DRESS histogram of a graph.
//
// Enumerates all C(N, k) subsets of k vertices.  For each subset,
// assigns weight `nabla_weight` to every edge incident to any vertex
// in the subset (all other edges keep their original weight), runs
// DRESS on the *same* graph topology, and accumulates every converged
// edge value into a single histogram.
//
// This is the individualization analogue of Δ^k-DRESS: same DFS over
// C(N,k) combinations, but instead of building a vertex-deleted
// subgraph, the weights are anchored to the target k-tuple.
// The topology (CSR adjacency, intercepts) is built once externally;
// only the edge_weight array is patched per combination.
//
// Parameters:
//   g          - input dress graph.  The graph's edge_weight, edge_dress,
//                edge_dress_next, and node_dress arrays are modified
//                in-place during each round but restored before return.
//   k          - individualization depth: number of vertices anchored
//                per subset.  k = 0 runs DRESS on the original graph.
//   iterations - maximum DRESS iterations per individualization.
//   epsilon    - convergence tolerance for DRESS and histogram bin width.
//                The histogram has floor(2 / epsilon) + 1 bins.
//   nabla_weight - multiplicative factor applied to edges incident to the
//                nabla vertices.  For each such edge e,
//                w(e) becomes w(e) * nabla_weight.  Non-incident edges
//                keep their original weight.  Any value != 1.0 breaks
//                symmetry; the default is 2.0.
//   hist_size  - [out] if non-NULL, set to floor(2 / epsilon) + 1.
//   keep_multisets - if non-zero, allocate and fill a flat matrix of
//                per-subset edge DRESS values.
//   multisets  - [out] on return, *multisets points to a heap-allocated
//                flat array of size C(N,k) * E.  Row s, edge e is at
//                (*multisets)[s * E + e].
//                The caller must free(*multisets).  Ignored (and set to
//                NULL) when keep_multisets is 0.  May be NULL itself.
//   num_subsets - [out] if non-NULL, set to C(N,k) on return.
//
// Returns:
//   A heap-allocated int64_t array of length floor(2 / epsilon) + 1.
//   hist[i] counts the number of converged edge values falling in bin
//   i = floor(d / epsilon).  The caller must free() the returned pointer.
//
// Complexity:
//   O( C(N,k) * iterations * E * d_max )
//   Graph construction cost is O(1) amortised (done once externally).
int64_t *nabla_fit(p_dress_graph_t g, int k, int iterations,
                  double epsilon, double nabla_weight,
                  int *hist_size,
                  int keep_multisets, double **multisets,
                  int64_t *num_subsets);

#ifdef __cplusplus
}
#endif

#endif // NABLA_DRESS_H
