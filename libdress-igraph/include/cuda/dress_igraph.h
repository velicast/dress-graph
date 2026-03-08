/**
 * cuda/dress_igraph.h — GPU-accelerated DRESS igraph wrapper via include-based switching.
 *
 * Drop-in replacement for dress_igraph.h.
 * Including this header redirects dress_fit_igraph() and delta_dress_fit_igraph()
 * to their CUDA implementations — no source changes required:
 *
 *   // CPU
 *   #include "dress_igraph.h"
 *   dress_fit_igraph(&graph, NULL, DRESS_VARIANT_UNDIRECTED, 100, 1e-6, 1, &result);
 *
 *   // CUDA — same call, different include
 *   #include "cuda/dress_igraph.h"
 *   dress_fit_igraph(&graph, NULL, DRESS_VARIANT_UNDIRECTED, 100, 1e-6, 1, &result);
 *
 * Do not include both this header and dress_igraph.h in the same
 * translation unit — the macros will conflict.
 */

#ifndef DRESS_IGRAPH_CUDA_REDIRECT_H
#define DRESS_IGRAPH_CUDA_REDIRECT_H

#include "dress_igraph.h"
#include "dress/cuda/dress.h"

/* Redirect CPU igraph symbols to CUDA implementations.
 * The underlying dress_fit / delta_dress_fit calls are already redirected
 * by dress/cuda/dress.h — so dress_fit_igraph() will transparently use
 * the CUDA kernels without any code changes. */
#define dress_fit_igraph       dress_fit_igraph_cuda
#define delta_dress_fit_igraph delta_dress_fit_igraph_cuda

#endif /* DRESS_IGRAPH_CUDA_REDIRECT_H */
