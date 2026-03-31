/**
 * dress/omp/igraph/dress.h — OpenMP-parallel DRESS igraph wrapper via
 *                             include-based switching.
 *
 * Drop-in replacement for dress/igraph/dress.h.
 * Including this header ensures dress_fit() and dress_delta_fit() use
 * the OpenMP backend — no source changes required:
 *
 *   // CPU
 *   #include <dress/igraph/dress.h>
 *   dress_fit(&graph, NULL, NULL, DRESS_VARIANT_UNDIRECTED, 100, 1e-6, 1, &result);
 *
 *   // OpenMP — same call, different include
 *   #include <dress/omp/igraph/dress.h>
 *   dress_fit(&graph, NULL, NULL, DRESS_VARIANT_UNDIRECTED, 100, 1e-6, 1, &result);
 *
 * Do not include both this header and dress/igraph/dress.h in the same
 * translation unit — the macros will conflict.
 */

#ifndef DRESS_IGRAPH_OMP_REDIRECT_H
#define DRESS_IGRAPH_OMP_REDIRECT_H

/* Pull in the OMP core redirect — sets guard macros. */
#include <dress/omp/dress.h>

/* Undo core OMP macros — the igraph wrapper resolves the backend
   through linking, not source-level macro replacement. */
#undef dress_fit
#undef dress_delta_fit
#undef dress_delta_fit_strided

/* The igraph base header declares the _igraph functions and sets up
   convenience macros: dress_fit → dress_fit_igraph, etc. */
#include <dress/igraph/dress.h>

#endif /* DRESS_IGRAPH_OMP_REDIRECT_H */
