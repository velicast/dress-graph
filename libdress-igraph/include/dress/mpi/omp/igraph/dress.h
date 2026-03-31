/**
 * dress/mpi/omp/igraph/dress.h — MPI + OpenMP distributed DRESS igraph
 *                                  wrapper via include-based switching.
 *
 * Drop-in replacement for dress/igraph/dress.h.
 * Including this single header redirects dress_fit() to OpenMP and
 * dress_delta_fit() to the OpenMP + MPI backend — no source changes required.
 *
 * Equivalent to:
 *   #include <dress/omp/igraph/dress.h>
 *   #include <dress/mpi/igraph/dress.h>
 *
 * Do not include dress/igraph/dress.h or dress/omp/igraph/dress.h
 * separately in the same translation unit — the macros will conflict.
 */
#ifndef DRESS_IGRAPH_MPI_OMP_REDIRECT_H
#define DRESS_IGRAPH_MPI_OMP_REDIRECT_H

/* Order matters: OMP first, then MPI detects it and routes to OMP + MPI. */
#include <dress/omp/igraph/dress.h>
#include <dress/mpi/igraph/dress.h>

#endif /* DRESS_IGRAPH_MPI_OMP_REDIRECT_H */
