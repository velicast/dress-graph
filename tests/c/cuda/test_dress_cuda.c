/**
 * test_dress_cuda.c — Compare GPU dress_fit_cuda() against CPU dress_fit()
 *                     on the same graph to verify numerical agreement.
 *
 * Usage:  ./test_dress_cuda
 *
 * Builds a small test graph, runs both CPU and GPU fit, and prints
 * the max absolute difference across all edge_dress and node_dress
 * values.  Exits 0 on success (diff < 1e-12), 1 otherwise.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dress/dress.h"
#include "dress/cuda/dress_cuda.h"

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/** Deep-copy a dress graph so we can run CPU and GPU independently. */
static p_dress_graph_t copy_dress_graph(p_dress_graph_t src)
{
    p_dress_graph_t g = (p_dress_graph_t)malloc(sizeof(dress_graph_t));
    int N = src->N, E = src->E;
    int S = src->adj_offset[N];

    g->N = N;
    g->E = E;
    g->variant = src->variant;
    g->precompute_intercepts = src->precompute_intercepts;

    /* U, V arrays */
    g->U = (int *)malloc(E * sizeof(int));
    g->V = (int *)malloc(E * sizeof(int));
    memcpy(g->U, src->U, E * sizeof(int));
    memcpy(g->V, src->V, E * sizeof(int));

    /* W (may be NULL) */
    if (src->W) {
        g->W = (double *)malloc(E * sizeof(double));
        memcpy(g->W, src->W, E * sizeof(double));
    } else {
        g->W = NULL;
    }

    /* CSR adjacency */
    g->adj_offset   = (int *)malloc((N + 1) * sizeof(int));
    g->adj_target   = (int *)malloc(S * sizeof(int));
    g->adj_edge_idx = (int *)malloc(S * sizeof(int));
    memcpy(g->adj_offset,   src->adj_offset,   (N + 1) * sizeof(int));
    memcpy(g->adj_target,   src->adj_target,   S * sizeof(int));
    memcpy(g->adj_edge_idx, src->adj_edge_idx, S * sizeof(int));

    /* Edge arrays */
    g->edge_weight     = (double *)malloc(E * sizeof(double));
    g->edge_dress      = (double *)malloc(E * sizeof(double));
    g->edge_dress_next = (double *)malloc(E * sizeof(double));
    memcpy(g->edge_weight,     src->edge_weight,     E * sizeof(double));
    memcpy(g->edge_dress,      src->edge_dress,      E * sizeof(double));
    memcpy(g->edge_dress_next, src->edge_dress_next, E * sizeof(double));

    /* Node array */
    g->node_dress = (double *)malloc(N * sizeof(double));
    memcpy(g->node_dress, src->node_dress, N * sizeof(double));

    /* Intercepts */
    if (src->precompute_intercepts) {
        int T = src->intercept_offset[E];
        g->intercept_offset = (int *)malloc((E + 1) * sizeof(int));
        memcpy(g->intercept_offset, src->intercept_offset, (E + 1) * sizeof(int));
        g->intercept_edge_ux = (int *)malloc(T * sizeof(int));
        g->intercept_edge_vx = (int *)malloc(T * sizeof(int));
        memcpy(g->intercept_edge_ux, src->intercept_edge_ux, T * sizeof(int));
        memcpy(g->intercept_edge_vx, src->intercept_edge_vx, T * sizeof(int));
    } else {
        g->intercept_offset  = NULL;
        g->intercept_edge_ux = NULL;
        g->intercept_edge_vx = NULL;
    }

    return g;
}

/** Build a simple triangle graph: 0-1, 1-2, 0-2 (undirected input). */
static void make_triangle(int *N_out, int *E_out,
                          int **U_out, int **V_out, double **W_out)
{
    *N_out = 3;
    *E_out = 3;
    *U_out = (int *)malloc(3 * sizeof(int));
    *V_out = (int *)malloc(3 * sizeof(int));
    *W_out = NULL; /* unweighted */
    int U[] = {0, 1, 0};
    int V[] = {1, 2, 2};
    memcpy(*U_out, U, 3 * sizeof(int));
    memcpy(*V_out, V, 3 * sizeof(int));
}

/** Build a small "house" graph (5 nodes, 6 edges). */
static void make_house(int *N_out, int *E_out,
                       int **U_out, int **V_out, double **W_out)
{
    *N_out = 5;
    *E_out = 6;
    *U_out = (int *)malloc(6 * sizeof(int));
    *V_out = (int *)malloc(6 * sizeof(int));
    *W_out = NULL;
    int U[] = {0,0,1,1,2,3};
    int V[] = {1,3,2,4,3,4};
    memcpy(*U_out, U, 6 * sizeof(int));
    memcpy(*V_out, V, 6 * sizeof(int));
}

/** Build a weighted star graph (4 nodes, 3 edges). */
static void make_weighted_star(int *N_out, int *E_out,
                               int **U_out, int **V_out, double **W_out)
{
    *N_out = 4;
    *E_out = 3;
    *U_out = (int *)malloc(3 * sizeof(int));
    *V_out = (int *)malloc(3 * sizeof(int));
    *W_out = (double *)malloc(3 * sizeof(double));
    int U[] = {0, 0, 0};
    int V[] = {1, 2, 3};
    double W[] = {1.0, 2.0, 3.0};
    memcpy(*U_out, U, 3 * sizeof(int));
    memcpy(*V_out, V, 3 * sizeof(int));
    memcpy(*W_out, W, 3 * sizeof(double));
}

/* ------------------------------------------------------------------ */
/*  Compare CPU vs GPU                                                 */
/* ------------------------------------------------------------------ */

static int run_test(const char *name, dress_variant_t variant,
                    int precompute_intercepts,
                    void (*builder)(int*, int*, int**, int**, double**))
{
    int N, E;
    int *U, *V;
    double *W;
    builder(&N, &E, &U, &V, &W);

    /* Build reference graph (CPU) */
    p_dress_graph_t g_cpu = dress_init_graph(N, E, U, V, W, NULL, variant,
                                             precompute_intercepts);

    /* Deep-copy for GPU run */
    p_dress_graph_t g_gpu = copy_dress_graph(g_cpu);

    int max_iter = 100;
    double eps   = 1e-15;
    int iter_cpu, iter_gpu;
    double delta_cpu, delta_gpu;

    /* Run CPU fit */
    dress_fit(g_cpu, max_iter, eps, &iter_cpu, &delta_cpu);

    /* Run GPU fit */
    dress_fit_cuda(g_gpu, max_iter, eps, &iter_gpu, &delta_gpu);

    /* Compare edge_dress */
    double max_diff_edge = 0.0;
    for (int e = 0; e < E; e++) {
        double d = fabs(g_cpu->edge_dress[e] - g_gpu->edge_dress[e]);
        if (d > max_diff_edge) max_diff_edge = d;
    }

    /* Compare node_dress */
    double max_diff_node = 0.0;
    for (int u = 0; u < N; u++) {
        double d = fabs(g_cpu->node_dress[u] - g_gpu->node_dress[u]);
        if (d > max_diff_node) max_diff_node = d;
    }

    double tol = 1e-12;
    int pass = (max_diff_edge < tol && max_diff_node < tol);

    printf("%-40s  iters=%d/%d  edge_diff=%.2e  node_diff=%.2e  %s\n",
           name, iter_cpu, iter_gpu, max_diff_edge, max_diff_node,
           pass ? "PASS" : "FAIL");

    dress_free_graph(g_cpu);
    dress_free_graph(g_gpu);

    return pass ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    int fail = 0;

    printf("=== DRESS CUDA vs CPU comparison ===\n\n");

    /* Triangle — all 4 variants × {with, without} intercepts */
    fail += run_test("triangle / undirected / no-intercepts",
                     DRESS_VARIANT_UNDIRECTED, 0, make_triangle);
    fail += run_test("triangle / undirected / intercepts",
                     DRESS_VARIANT_UNDIRECTED, 1, make_triangle);
    fail += run_test("triangle / directed / no-intercepts",
                     DRESS_VARIANT_DIRECTED, 0, make_triangle);
    fail += run_test("triangle / directed / intercepts",
                     DRESS_VARIANT_DIRECTED, 1, make_triangle);
    fail += run_test("triangle / forward / no-intercepts",
                     DRESS_VARIANT_FORWARD, 0, make_triangle);
    fail += run_test("triangle / forward / intercepts",
                     DRESS_VARIANT_FORWARD, 1, make_triangle);
    fail += run_test("triangle / backward / no-intercepts",
                     DRESS_VARIANT_BACKWARD, 0, make_triangle);
    fail += run_test("triangle / backward / intercepts",
                     DRESS_VARIANT_BACKWARD, 1, make_triangle);

    /* House graph */
    fail += run_test("house / undirected / no-intercepts",
                     DRESS_VARIANT_UNDIRECTED, 0, make_house);
    fail += run_test("house / undirected / intercepts",
                     DRESS_VARIANT_UNDIRECTED, 1, make_house);
    fail += run_test("house / directed / intercepts",
                     DRESS_VARIANT_DIRECTED, 1, make_house);
    fail += run_test("house / forward / intercepts",
                     DRESS_VARIANT_FORWARD, 1, make_house);

    /* Weighted star */
    fail += run_test("weighted-star / undirected / no-intercepts",
                     DRESS_VARIANT_UNDIRECTED, 0, make_weighted_star);
    fail += run_test("weighted-star / undirected / intercepts",
                     DRESS_VARIANT_UNDIRECTED, 1, make_weighted_star);
    fail += run_test("weighted-star / directed / intercepts",
                     DRESS_VARIANT_DIRECTED, 1, make_weighted_star);

    printf("\n%s (%d failures)\n", fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", fail);
    return fail ? 1 : 0;
}
