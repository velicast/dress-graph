/**
 * delta_dress_cuda.c — GPU-accelerated Δ^k-DRESS histogram.
 *
 * Delegates to the shared delta_dress_impl.h, passing dress_fit_cuda
 * as the fitting function.
 */

#ifdef DRESS_CUDA

#include "dress/cuda/dress_cuda.h"
#include "delta_dress_impl.h"

int64_t *delta_dress_fit_cuda(p_dress_graph_t g, int k, int iterations,
                              double epsilon, int *hist_size,
                              int keep_multisets, double **multisets,
                              int64_t *num_subgraphs)
{
    return delta_dress_fit_impl(g, k, iterations, epsilon, hist_size,
                                keep_multisets, multisets, num_subgraphs,
                                dress_fit_cuda, 0, 1);
}

int64_t *delta_dress_fit_cuda_strided(p_dress_graph_t g, int k,
                                      int iterations, double epsilon,
                                      int *hist_size,
                                      int keep_multisets, double **multisets,
                                      int64_t *num_subgraphs,
                                      int offset, int stride)
{
    return delta_dress_fit_impl(g, k, iterations, epsilon, hist_size,
                                keep_multisets, multisets, num_subgraphs,
                                dress_fit_cuda, offset, stride);
}

#endif /* DRESS_CUDA */

/* Prevent -Wempty-translation-unit when CUDA is unavailable. */
typedef int delta_dress_cuda_unused_t;
