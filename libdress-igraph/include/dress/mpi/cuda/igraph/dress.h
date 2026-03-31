/**
 * dress/mpi/cuda/igraph/dress.h — MPI + CUDA distributed DRESS igraph
 *                                  wrapper via include-based switching.
 *
 * Drop-in replacement for dress/igraph/dress.h.
 * Including this single header redirects dress_delta_fit() to
 * the CUDA + MPI backend — no source changes required.
 *
 * Equivalent to:
 *   #include <dress/cuda/igraph/dress.h>
 *   #include <dress/mpi/igraph/dress.h>
 *
 * Do not include dress/igraph/dress.h or dress/cuda/igraph/dress.h
 * separately in the same translation unit — the macros will conflict.
 */
#ifndef DRESS_IGRAPH_MPI_CUDA_REDIRECT_H
#define DRESS_IGRAPH_MPI_CUDA_REDIRECT_H

/* Order matters: CUDA first, then MPI detects it and routes to GPU + MPI. */
#include <dress/cuda/igraph/dress.h>
#include <dress/mpi/igraph/dress.h>

#endif /* DRESS_IGRAPH_MPI_CUDA_REDIRECT_H */
