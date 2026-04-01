/*
 * dress_igraph.c — igraph wrapper for the dress edge similarity algorithm.
 *
 * Bridges igraph_t graphs to the DRESS C API by extracting the edge list,
 * optional edge weights, running the iterative dress fitting, and exposing
 * results as zero-copy const pointers into the internal dress_graph_t.
 *
 * Build:
 *   gcc -O3 -c dress_igraph.c -o dress_igraph.o $(pkg-config --cflags igraph)
 *
 * Link with DRESS.o and igraph:
 *   gcc main.o dress_igraph.o DRESS.o -o main $(pkg-config --libs igraph) -lm -fopenmp
 */

#include <dress/igraph/dress.h>
#include "dress/dress.h"

/* Undo convenience macros — this file implements the igraph wrapper
   and calls the core dress_fit() / dress_delta_fit() directly. */
#undef dress_fit
#undef dress_delta_fit

#include <igraph/igraph.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  dress_fit_igraph                                               */
/* ------------------------------------------------------------------ */

int dress_fit_igraph(const igraph_t *graph,
                         const char *weight_attr,
                         const char *vertex_weight_attr,
                         dress_variant_t variant,
                         int max_iters,
                         double epsilon,
                         int precompute,
                         dress_result_igraph_t *result)
{
    if (!graph || !result)
        return -1;

    int N = (int)igraph_vcount(graph);
    int E = (int)igraph_ecount(graph);

    if (E == 0) {
        memset(result, 0, sizeof(*result));
        result->N = N;
        return 0;
    }

    /* ----- Allocate edge arrays (ownership transferred to DRESS) ----- */

    int    *U = (int *)   malloc(E * sizeof(int));
    int    *V = (int *)   malloc(E * sizeof(int));
    double *W = NULL;

    if (!U || !V) {
        free(U); free(V);
        return -1;
    }

    /* ----- Extract edge list from igraph ----- */

    for (int e = 0; e < E; e++) {
        U[e] = (int)IGRAPH_FROM(graph, e);
        V[e] = (int)IGRAPH_TO(graph, e);
    }

    /* ----- Extract edge weights (optional) ----- */

    if (weight_attr != NULL &&
        igraph_cattribute_has_attr(graph, IGRAPH_ATTRIBUTE_EDGE, weight_attr))
    {
        W = (double *)malloc(E * sizeof(double));
        if (!W) {
            free(U); free(V);
            return -1;
        }
        for (int e = 0; e < E; e++) {
            W[e] = igraph_cattribute_EAN(graph, weight_attr, e);
        }
    }

    /* ----- Extract vertex weights (optional) ----- */

    double *NW = NULL;
    if (vertex_weight_attr != NULL &&
        igraph_cattribute_has_attr(graph, IGRAPH_ATTRIBUTE_VERTEX, vertex_weight_attr))
    {
        NW = (double *)malloc(N * sizeof(double));
        if (!NW) {
            free(U); free(V); free(W);
            return -1;
        }
        for (int v = 0; v < N; v++) {
            NW[v] = igraph_cattribute_VAN(graph, vertex_weight_attr, v);
        }
    }

    /* ----- Build DRESS graph and fit ----- */
    /*
     * dress_init_graph takes ownership of U, V, W, NW.
     * They will be freed by dress_free_graph.
     */

    p_dress_graph_t dg = dress_init_graph(N, E, U, V, W, NW,
                                          variant, precompute);
    if (!dg)
        return -1;

    int    iterations = 0;
    double delta      = 0.0;

    dress_fit(dg, max_iters, epsilon, &iterations, &delta);

    /* ----- Point directly into DRESS arrays (zero-copy) ----- */
    /*
     * The dress_graph_t owns all the arrays.  We keep it alive inside
     * the result struct and just expose const pointers.  No memcpy.
     * dress_free_igraph() will call dress_free_graph() later.
     */

    result->N          = N;
    result->E          = E;
    result->iterations = iterations;
    result->delta      = delta;
    result->src        = dg->U;
    result->dst        = dg->V;
    result->dress      = dg->edge_dress;
    result->weight     = dg->edge_weight;
    result->vertex_dress = dg->vertex_dress;
    result->dg_        = dg;

    return 0;
}

/* ------------------------------------------------------------------ */
/*  dress_free_igraph                                                  */
/* ------------------------------------------------------------------ */

void dress_free_igraph(dress_result_igraph_t *result)
{
    if (!result) return;

    if (result->dg_)
        dress_free_graph(result->dg_);

    memset(result, 0, sizeof(*result));
}

/* ------------------------------------------------------------------ */
/*  dress_to_vector_igraph                                             */
/* ------------------------------------------------------------------ */

int dress_to_vector_igraph(const dress_result_igraph_t *result,
                           igraph_vector_t *out)
{
    if (!result || !out)
        return -1;

    igraph_vector_resize(out, result->E);

    for (int e = 0; e < result->E; e++)
        VECTOR(*out)[e] = result->dress[e];

    return 0;
}

/* ================================================================== */
/*  Δ^k-DRESS igraph wrapper                                           */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/*  dress_delta_fit_igraph                                         */
/* ------------------------------------------------------------------ */

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
                           delta_dress_result_igraph_t *result)
{
    if (!graph || !result)
        return -1;

    int N = (int)igraph_vcount(graph);
    int E = (int)igraph_ecount(graph);

    memset(result, 0, sizeof(*result));

    if (E == 0)
        return 0;

    /* ----- Allocate edge arrays (ownership transferred to DRESS) ----- */

    int    *U = (int *)   malloc(E * sizeof(int));
    int    *V = (int *)   malloc(E * sizeof(int));
    double *W = NULL;

    if (!U || !V) {
        free(U); free(V);
        return -1;
    }

    /* ----- Extract edge list from igraph ----- */

    for (int e = 0; e < E; e++) {
        U[e] = (int)IGRAPH_FROM(graph, e);
        V[e] = (int)IGRAPH_TO(graph, e);
    }

    /* ----- Extract edge weights (optional) ----- */

    if (weight_attr != NULL &&
        igraph_cattribute_has_attr(graph, IGRAPH_ATTRIBUTE_EDGE, weight_attr))
    {
        W = (double *)malloc(E * sizeof(double));
        if (!W) {
            free(U); free(V);
            return -1;
        }
        for (int e = 0; e < E; e++) {
            W[e] = igraph_cattribute_EAN(graph, weight_attr, e);
        }
    }

    /* ----- Extract vertex weights (optional) ----- */

    double *NW = NULL;
    if (vertex_weight_attr != NULL &&
        igraph_cattribute_has_attr(graph, IGRAPH_ATTRIBUTE_VERTEX, vertex_weight_attr))
    {
        NW = (double *)malloc(N * sizeof(double));
        if (!NW) {
            free(U); free(V); free(W);
            return -1;
        }
        for (int v = 0; v < N; v++) {
            NW[v] = igraph_cattribute_VAN(graph, vertex_weight_attr, v);
        }
    }

    /* ----- Build DRESS graph and run delta_fit ----- */

    p_dress_graph_t dg = dress_init_graph(N, E, U, V, W, NW,
                                          variant, precompute);
    if (!dg)
        return -1;

    int hist_size = 0;
    double *ms_ptr = NULL;
    int64_t num_sub = 0;
    dress_hist_pair_t *histogram = dress_delta_fit(dg, k, max_iters, epsilon,
                                               n_samples, seed,
                                               compute_histogram ? &hist_size : NULL,
                                               keep_multisets,
                                               keep_multisets ? &ms_ptr : NULL,
                                               &num_sub);

    dress_free_graph(dg);

    result->histogram = histogram;
    result->hist_size = hist_size;
    result->multisets = ms_ptr;
    result->num_subgraphs = num_sub;

    return 0;
}

/* ------------------------------------------------------------------ */
/*  delta_dress_free_igraph                                            */
/* ------------------------------------------------------------------ */

void delta_dress_free_igraph(delta_dress_result_igraph_t *result)
{
    if (!result) return;

    free(result->histogram);
    free(result->multisets);
    memset(result, 0, sizeof(*result));
}

/* ------------------------------------------------------------------ */
/*  delta_dress_to_vector_igraph                                       */
/* ------------------------------------------------------------------ */

int delta_dress_to_vector_igraph(const delta_dress_result_igraph_t *result,
                                 igraph_vector_t *out)
{
    if (!result || !out)
        return -1;

    igraph_vector_resize(out, 2 * result->hist_size);

    for (int i = 0; i < result->hist_size; i++) {
        VECTOR(*out)[2 * i] = result->histogram[i].value;
        VECTOR(*out)[2 * i + 1] = (double)result->histogram[i].count;
    }

    return 0;
}
