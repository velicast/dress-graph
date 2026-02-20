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

#include "dress_igraph.h"
#include "dress/dress.h"

#include <igraph/igraph.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  dress_igraph_compute                                               */
/* ------------------------------------------------------------------ */

int dress_igraph_compute(const igraph_t *graph,
                         const char *weight_attr,
                         dress_variant_t variant,
                         int max_iters,
                         double epsilon,
                         int precompute,
                         dress_igraph_result_t *result)
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

    /* ----- Build DRESS graph and fit ----- */
    /*
     * init_dress_graph takes ownership of U, V, W.
     * They will be freed by free_dress_graph.
     */

    p_dress_graph_t dg = init_dress_graph(N, E, U, V, W,
                                          variant, precompute);
    if (!dg)
        return -1;

    int    iterations = 0;
    double delta      = 0.0;

    fit(dg, max_iters, epsilon, &iterations, &delta);

    /* ----- Point directly into DRESS arrays (zero-copy) ----- */
    /*
     * The dress_graph_t owns all the arrays.  We keep it alive inside
     * the result struct and just expose const pointers.  No memcpy.
     * dress_igraph_free() will call free_dress_graph() later.
     */

    result->N          = N;
    result->E          = E;
    result->iterations = iterations;
    result->delta      = delta;
    result->src        = dg->U;
    result->dst        = dg->V;
    result->dress      = dg->edge_dress;
    result->weight     = dg->edge_weight;
    result->node_dress = dg->node_dress;
    result->dg_        = dg;

    return 0;
}

/* ------------------------------------------------------------------ */
/*  dress_igraph_free                                                  */
/* ------------------------------------------------------------------ */

void dress_igraph_free(dress_igraph_result_t *result)
{
    if (!result) return;

    if (result->dg_)
        free_dress_graph(result->dg_);

    memset(result, 0, sizeof(*result));
}

/* ------------------------------------------------------------------ */
/*  dress_igraph_to_vector                                             */
/* ------------------------------------------------------------------ */

int dress_igraph_to_vector(const dress_igraph_result_t *result,
                           igraph_vector_t *out)
{
    if (!result || !out)
        return -1;

    igraph_vector_resize(out, result->E);

    for (int e = 0; e < result->E; e++)
        VECTOR(*out)[e] = result->dress[e];

    return 0;
}
