/*
 * dress_igraph.h — igraph wrapper for the dress edge similarity algorithm.
 *
 * Provides a simple bridge between igraph_t graphs and the DRESS C API.
 *
 * Usage:
 *   igraph_t graph;
 *   igraph_read_graph_edgelist(&graph, fp, 0, IGRAPH_UNDIRECTED);
 *
 *   dress_igraph_result_t result;
 *   dress_igraph_compute(&graph, NULL,         // NULL = unweighted
 *                        DRESS_VARIANT_UNDIRECTED,
 *                        100, 1e-6, 1,          // maxIters, eps, precompute
 *                        &result);
 *
 *   for (int e = 0; e < result.E; e++)
 *       printf("%d %d %.6f\n", result.src[e], result.dst[e], result.dress[e]);
 *
 *   dress_igraph_free(&result);
 *   igraph_destroy(&graph);
 */
#ifndef DRESS_IGRAPH_H
#define DRESS_IGRAPH_H

#include "dress/dress.h"
#include <igraph/igraph.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Result structure                                                   */
/* ------------------------------------------------------------------ */

typedef struct dress_igraph_result_t {
    int     N;           /* number of vertices                        */
    int     E;           /* number of edges                           */
    const int    *src;   /* [E] edge source endpoints (owned by dg_)  */
    const int    *dst;   /* [E] edge target endpoints (owned by dg_)  */
    const double *dress; /* [E] dress similarity per edge             */
    const double *weight;/* [E] variant edge weight                   */
    const double *node_dress; /* [N] per-node dress norm              */
    int     iterations;  /* iterations performed                      */
    double  delta;       /* final max-delta at convergence            */
    p_dress_graph_t dg_; /* internal — do not access directly         */
} dress_igraph_result_t;

/* ------------------------------------------------------------------ */
/*  API                                                                */
/* ------------------------------------------------------------------ */

/*
 * dress_igraph_compute
 *
 * Run dress on an igraph graph and write the results into `result`.
 *
 * Parameters:
 *   graph        — pointer to a valid igraph_t (not modified)
 *   weight_attr  — name of an edge attribute holding weights, or NULL
 *                  for unweighted graphs. If the attribute does not
 *                  exist on an edge, weight 1.0 is used.
 *   variant      — DRESS_VARIANT_UNDIRECTED / DIRECTED / FORWARD / BACKWARD
 *   max_iters    — maximum fitting iterations
 *   epsilon      — convergence threshold
 *   precompute   — 1 to precompute neighborhood intercepts, 0 otherwise
 *   result       — output struct (caller-allocated, contents filled in)
 *
 * Returns 0 on success, non-zero on error.
 */
int dress_igraph_compute(const igraph_t *graph,
                         const char *weight_attr,
                         dress_variant_t variant,
                         int max_iters,
                         double epsilon,
                         int precompute,
                         dress_igraph_result_t *result);

/*
 * dress_igraph_free
 *
 * Free all heap memory inside a dress_igraph_result_t.
 * The struct itself is not freed (it may be stack-allocated).
 */
void dress_igraph_free(dress_igraph_result_t *result);

/*
 * dress_igraph_to_vector
 *
 * Copy the per-edge dress values from a result into an igraph_vector_t,
 * ordered by igraph edge id [0..E-1].  The igraph_vector_t must be
 * initialized by the caller.
 */
int dress_igraph_to_vector(const dress_igraph_result_t *result,
                           igraph_vector_t *out);

#ifdef __cplusplus
}
#endif

#endif /* DRESS_IGRAPH_H */
