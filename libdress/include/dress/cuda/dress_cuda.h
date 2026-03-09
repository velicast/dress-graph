/**
 * dress_cuda.h — GPU-accelerated dress_fit() for libdress.
 *
 * Drop-in replacement for the CPU dress_fit():
 *
 *   #include "dress/dress.h"
 *   #include "dress/cuda/dress_cuda.h"
 *
 *   p_dress_graph_t g = init_dress_graph(N, E, U, V, W, variant, precompute);
 *   dress_fit_cuda(g, max_iterations, epsilon, &iterations, &delta);
 *   // g->edge_dress and g->node_dress are now populated
 *   free_dress_graph(g);
 *
 * Graph construction (init_dress_graph) stays on the CPU.
 * Only the iterative fitting loop runs on the GPU.
 */

#ifndef DRESS_CUDA_H
#define DRESS_CUDA_H

#include "dress/dress.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GPU-accelerated iterative DRESS fitting.
 *
 * Same signature and semantics as the CPU dress_fit() in dress.h.
 * Uploads arrays to GPU, runs the iteration loop with CUDA kernels,
 * downloads edge_dress and node_dress back to the host.
 *
 * Supports all four variants (UNDIRECTED, DIRECTED, FORWARD, BACKWARD)
 * and both intercept / non-intercept code paths.
 */
void dress_fit_cuda(p_dress_graph_t g, int max_iterations, double epsilon,
                   int *iterations, double *delta);

/**
 * GPU-accelerated Δ^k-DRESS histogram computation.
 *
 * Same interface as the CPU delta_dress_fit() in delta_dress.h, but each
 * subgraph is fitted on the GPU via dress_fit_cuda().
 */
int64_t *delta_dress_fit_cuda(p_dress_graph_t g, int k, int iterations,
                              double epsilon, int *hist_size,
                              int keep_multisets, double **multisets,
                              int64_t *num_subgraphs);

/**
 * GPU-accelerated strided Δ^k-DRESS for distributed computation.
 *
 * Same as delta_dress_fit_cuda but processes only subgraphs where
 * index % stride == offset.  With offset=0, stride=1 processes all.
 */
int64_t *delta_dress_fit_cuda_strided(p_dress_graph_t g, int k,
                                      int iterations, double epsilon,
                                      int *hist_size,
                                      int keep_multisets, double **multisets,
                                      int64_t *num_subgraphs,
                                      int offset, int stride);

#ifdef __cplusplus
}
#endif

#endif /* DRESS_CUDA_H */
