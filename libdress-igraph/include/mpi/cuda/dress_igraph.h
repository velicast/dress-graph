/**
 * mpi/cuda/dress_igraph.h — MPI + CUDA distributed DRESS igraph wrapper
 *                           via include-based switching.
 *
 * Drop-in replacement for dress_igraph.h.
 * Including this single header redirects delta_dress_fit_igraph() to
 * the CUDA + MPI backend — no source changes required.
 *
 * Equivalent to:
 *   #include "cuda/dress_igraph.h"
 *   #include "mpi/dress_igraph.h"
 *
 * Do not include dress_igraph.h or cuda/dress_igraph.h separately in the
 * same translation unit — the macros will conflict.
 */
#ifndef DRESS_IGRAPH_MPI_CUDA_REDIRECT_H
#define DRESS_IGRAPH_MPI_CUDA_REDIRECT_H

/* Order matters: CUDA first, then MPI detects it and routes to GPU + MPI. */
#include "cuda/dress_igraph.h"
#include "mpi/dress_igraph.h"

#endif /* DRESS_IGRAPH_MPI_CUDA_REDIRECT_H */
