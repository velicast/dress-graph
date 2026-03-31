/**
 * dress/mpi/cuda/dress.h — MPI + CUDA distributed DRESS via
 *                           include-based switching.
 *
 * Drop-in replacement for dress/dress.h.
 * Including this single header redirects dress_fit() to CUDA and
 * dress_delta_fit() to the CUDA + MPI backend — no source changes required.
 *
 * Equivalent to:
 *   #include "dress/cuda/dress.h"
 *   #include "dress/mpi/dress.h"
 *
 * Do not include dress/dress.h or dress/cuda/dress.h separately in the
 * same translation unit — the macros will conflict.
 */
#ifndef DRESS_MPI_CUDA_REDIRECT_H
#define DRESS_MPI_CUDA_REDIRECT_H

/* Order matters: CUDA first, then MPI detects it and routes to GPU + MPI. */
#include "dress/cuda/dress.h"
#include "dress/mpi/dress.h"

#endif /* DRESS_MPI_CUDA_REDIRECT_H */
