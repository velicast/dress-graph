/*
 * dress/igraph/dress.h — igraph wrapper for the DRESS edge similarity algorithm.
 *
 * Provides a simple bridge between igraph_t graphs and the DRESS C API.
 *
 * Usage:
 *   igraph_t graph;
 *   igraph_read_graph_edgelist(&graph, fp, 0, IGRAPH_UNDIRECTED);
 *
 *   dress_result_igraph_t result;
 *   dress_fit(&graph, NULL,                     // NULL = unweighted
 *             DRESS_VARIANT_UNDIRECTED,
 *             100, 1e-6, 1,                     // maxIters, eps, precompute
 *             &result);
 *
 *   for (int e = 0; e < result.E; e++)
 *       printf("%d %d %.6f\n", result.src[e], result.dst[e], result.dress[e]);
 *
 *   dress_free(&result);
 *   igraph_destroy(&graph);
 */
#ifndef DRESS_IGRAPH_H
#define DRESS_IGRAPH_H

#include <dress/dress.h>
#include <igraph/igraph.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Result structures                                                  */
/* ------------------------------------------------------------------ */

typedef struct dress_result_igraph_t {
    int     N;           /* number of vertices                        */
    int     E;           /* number of edges                           */
    const int    *src;   /* [E] edge source endpoints (owned by dg_)  */
    const int    *dst;   /* [E] edge target endpoints (owned by dg_)  */
    const double *dress; /* [E] dress similarity per edge             */
    const double *weight;/* [E] variant edge weight                   */
    const double *vertex_dress; /* [N] per-vertex dress norm              */
    int     iterations;  /* iterations performed                      */
    double  delta;       /* final max-delta at convergence            */
    p_dress_graph_t dg_; /* internal — do not access directly         */
} dress_result_igraph_t;

/* ------------------------------------------------------------------ */
/*  API                                                                */
/* ------------------------------------------------------------------ */

/*
 * dress_fit_igraph
 *
 * Run DRESS on an igraph graph and write the results into `result`.
 *
 * Parameters:
 *   graph        — pointer to a valid igraph_t (not modified)
 *   weight_attr  — name of an edge attribute holding weights, or NULL
 *                  for unweighted graphs. If the attribute does not
 *                  exist on an edge, weight 1.0 is used.
 *   vertex_weight_attr — name of a vertex attribute holding vertex weights,
 *                  or NULL for uniform vertex weights (all 1.0).
 *   variant      — DRESS_VARIANT_UNDIRECTED / DIRECTED / FORWARD / BACKWARD
 *   max_iters    — maximum fitting iterations
 *   epsilon      — convergence threshold
 *   precompute   — 1 to precompute neighborhood intercepts, 0 otherwise
 *   result       — output struct (caller-allocated, contents filled in)
 *
 * Returns 0 on success, non-zero on error.
 */
int dress_fit_igraph(const igraph_t *graph,
                     const char *weight_attr,
                     const char *vertex_weight_attr,
                     dress_variant_t variant,
                     int max_iters,
                     double epsilon,
                     int precompute,
                     dress_result_igraph_t *result);

/*
 * dress_free_igraph
 *
 * Free all heap memory inside a dress_result_igraph_t.
 * The struct itself is not freed (it may be stack-allocated).
 */
void dress_free_igraph(dress_result_igraph_t *result);

/*
 * dress_to_vector_igraph
 *
 * Copy the per-edge dress values from a result into an igraph_vector_t,
 * ordered by igraph edge id [0..E-1].  The igraph_vector_t must be
 * initialized by the caller.
 */
int dress_to_vector_igraph(const dress_result_igraph_t *result,
                           igraph_vector_t *out);

/* ------------------------------------------------------------------ */
/*  Δ^k-DRESS result structure                                         */
/* ------------------------------------------------------------------ */

typedef struct delta_dress_result_igraph_t {
    dress_hist_pair_t *histogram;  /* [hist_size] sparse exact histogram entries */
    int                hist_size;  /* number of exact histogram entries */
    double            *multisets;  /* [num_subgraphs * E] row-major, NaN = removed; NULL when not requested */
    int64_t            num_subgraphs; /* C(N,k) */
} delta_dress_result_igraph_t;

/* ------------------------------------------------------------------ */
/*  Δ^k-DRESS API                                                      */
/* ------------------------------------------------------------------ */

/*
 * dress_delta_fit_igraph
 *
 * Run Δ^k-DRESS on an igraph graph and write results into `result`.
 *
 * Parameters:
 *   graph        — pointer to a valid igraph_t (not modified)
 *   weight_attr  — name of an edge attribute holding weights, or NULL
 *   vertex_weight_attr — name of a vertex attribute holding vertex weights,
 *                  or NULL for uniform vertex weights (all 1.0).
 *   variant      — DRESS_VARIANT_UNDIRECTED / DIRECTED / FORWARD / BACKWARD
 *   k            — deletion depth: vertices removed per subset
 *   max_iters    — maximum DRESS iterations per subgraph
 *   epsilon      — convergence threshold for the per-subgraph DRESS fits
 *   precompute   — 1 to precompute neighborhood intercepts, 0 otherwise
 *   result       — output struct (caller-allocated, contents filled in)
 *
 * Returns 0 on success, non-zero on error.
 */
int dress_delta_fit_igraph(const igraph_t *graph,
                           const char *weight_attr,
                           const char *vertex_weight_attr,
                           dress_variant_t variant,
                           int k,
                           int max_iters,
                           double epsilon,
                           int n_samples,
                           unsigned int seed,
                           int precompute,
                           int keep_multisets,
                           int compute_histogram,
                           delta_dress_result_igraph_t *result);

/*
 * delta_dress_free_igraph
 *
 * Free all heap memory inside a delta_dress_result_igraph_t.
 * The struct itself is not freed (it may be stack-allocated).
 */
void delta_dress_free_igraph(delta_dress_result_igraph_t *result);

/*
 * delta_dress_to_vector_igraph
 *
 * Copy the sparse exact histogram from a delta result into an
 * igraph_vector_t as interleaved value/count pairs:
 *   [value0, count0, value1, count1, ...]
 * The igraph_vector_t must be initialized by the caller.
 */
int delta_dress_to_vector_igraph(const delta_dress_result_igraph_t *result,
                                 igraph_vector_t *out);

#ifdef __cplusplus
}
#endif

/* ── Convenience macros — call dress_fit() / dress_delta_fit() etc.
      as with the core API; the igraph backend is transparent. ───── */

#define dress_fit             dress_fit_igraph
#define dress_free            dress_free_igraph
#define dress_to_vector       dress_to_vector_igraph
#define dress_delta_fit       dress_delta_fit_igraph
#define delta_dress_free      delta_dress_free_igraph
#define delta_dress_to_vector delta_dress_to_vector_igraph

#endif /* DRESS_IGRAPH_H */
