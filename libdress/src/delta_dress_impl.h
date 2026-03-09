/**
 * delta_dress_impl.h — internal declarations for the shared Δ^k-DRESS
 * implementation.
 *
 * The actual code lives in delta_dress_impl.c.  This header is included
 * by delta_dress.c and cuda/delta_dress_cuda.c so they can call
 * delta_dress_fit_impl() with different fit-function pointers.
 */

#ifndef DELTA_DRESS_IMPL_H
#define DELTA_DRESS_IMPL_H

#include "dress/dress.h"
#include <stdint.h>

/* Fit-function signature: same as dress_fit / dress_fit_cuda. */
typedef void (*dress_fit_fn)(p_dress_graph_t, int, double, int *, double *);

/**
 * Core Δ^k-DRESS: enumerate C(N,k) deletion subsets, fit each subgraph
 * using `fit_fn`, and accumulate the pooled histogram.
 *
 * offset/stride control which subgraphs to process:
 *   offset=0, stride=1  →  all subgraphs (default, non-MPI)
 *   offset=rank, stride=nprocs  →  round-robin MPI distribution
 */
int64_t *delta_dress_fit_impl(p_dress_graph_t g, int k,
                               int iterations, double epsilon,
                               int *hist_size,
                               int keep_multisets,
                               double **multisets,
                               int64_t *num_subgraphs,
                               dress_fit_fn fit_fn,
                               int offset, int stride);

#endif /* DELTA_DRESS_IMPL_H */
