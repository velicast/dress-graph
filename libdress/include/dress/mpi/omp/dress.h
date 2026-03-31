/**
 * dress/mpi/omp/dress.h — MPI + OpenMP distributed DRESS via
 *                          include-based switching.
 *
 * Drop-in replacement for dress/dress.h.
 * Including this single header redirects dress_fit() to OpenMP and
 * dress_delta_fit() to the OpenMP + MPI backend — no source changes required.
 *
 *   // Sequential
 *   #include "dress/dress.h"
 *   dress_fit(g, 100, 1e-6, &iters, &delta);
 *   dress_delta_fit(g, 2, 100, 1e-6, &hs, 0, NULL, &ns);
 *
 *   // MPI + OpenMP — same calls, different include
 *   #include "dress/mpi/omp/dress.h"
 *   dress_fit(g, 100, 1e-6, &iters, &delta);       // OpenMP edge-parallel
 *   dress_delta_fit(g, 2, 100, 1e-6, &hs, 0, NULL, &ns); // MPI + OMP
 *
 * Equivalent to:
 *   #include "dress/omp/dress.h"
 *   #include "dress/mpi/dress.h"
 *
 * Do not include dress/dress.h or dress/omp/dress.h separately in the
 * same translation unit — the macros will conflict.
 */
#ifndef DRESS_MPI_OMP_REDIRECT_H
#define DRESS_MPI_OMP_REDIRECT_H

/* Order matters: OMP first, then MPI detects it and routes to OMP + MPI. */
#include "dress/omp/dress.h"
#include "dress/mpi/dress.h"

#endif /* DRESS_MPI_OMP_REDIRECT_H */
