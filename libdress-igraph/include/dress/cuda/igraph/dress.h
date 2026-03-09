/**
 * dress/cuda/igraph/dress.h — GPU-accelerated DRESS igraph wrapper via
 *                              include-based switching.
 *
 * Drop-in replacement for dress/igraph/dress.h.
 * Including this header ensures dress_fit() and delta_dress_fit() use
 * the CUDA backend — no source changes required:
 *
 *   // CPU
 *   #include <dress/igraph/dress.h>
 *   dress_fit(&graph, NULL, DRESS_VARIANT_UNDIRECTED, 100, 1e-6, 1, &result);
 *
 *   // CUDA — same call, different include
 *   #include <dress/cuda/igraph/dress.h>
 *   dress_fit(&graph, NULL, DRESS_VARIANT_UNDIRECTED, 100, 1e-6, 1, &result);
 *
 * Do not include both this header and dress/igraph/dress.h in the same
 * translation unit — the macros will conflict.
 */

#ifndef DRESS_IGRAPH_CUDA_REDIRECT_H
#define DRESS_IGRAPH_CUDA_REDIRECT_H

/* Pull in the CUDA core redirect — sets guard macros that the MPI
   igraph header can detect later. */
#include <dress/cuda/dress.h>

/* Undo core CUDA macros — the igraph wrapper resolves the backend
   through linking, not source-level macro replacement. */
#undef dress_fit
#undef delta_dress_fit
#undef delta_dress_fit_strided

/* The igraph base header declares the _igraph functions and sets up
   convenience macros: dress_fit → dress_fit_igraph, etc. */
#include <dress/igraph/dress.h>

#endif /* DRESS_IGRAPH_CUDA_REDIRECT_H */
