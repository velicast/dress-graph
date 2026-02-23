/*
 * Tests for the DRESS C library (libdress).
 *
 * Build from the repo root:
 *   gcc -O2 -I libdress/include -o tests/c/test_dress \
 *       tests/c/test_dress.c libdress/src/dress.c -lm -fopenmp
 *
 * Run:
 *   ./tests/c/test_dress
 */

#include "dress/dress.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── tiny test harness ─────────────────────────────────────────────── */

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT(cond, msg)                                                \
    do {                                                                 \
        if (!(cond)) {                                                   \
            fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__,    \
                    (msg));                                               \
            g_fail++;                                                    \
        } else {                                                         \
            g_pass++;                                                    \
        }                                                                \
    } while (0)

#define ASSERT_EQ_INT(a, b, msg)    ASSERT((a) == (b), msg)
#define ASSERT_GT(a, b, msg)        ASSERT((a) >  (b), msg)
#define ASSERT_LT(a, b, msg)        ASSERT((a) <  (b), msg)
#define ASSERT_NEAR(a, b, tol, msg) ASSERT(fabs((a) - (b)) < (tol), msg)

/* ── helpers ───────────────────────────────────────────────────────── */

/* Allocate a malloc'd copy of a stack array (C API takes ownership). */
static int *dup_int(const int *src, int n)
{
    int *p = (int *)malloc((size_t)n * sizeof(int));
    memcpy(p, src, (size_t)n * sizeof(int));
    return p;
}

static double *dup_double(const double *src, int n)
{
    double *p = (double *)malloc((size_t)n * sizeof(double));
    memcpy(p, src, (size_t)n * sizeof(double));
    return p;
}

/* ── test: construction ────────────────────────────────────────────── */

static void test_unweighted_triangle(void)
{
    printf("test_unweighted_triangle\n");

    int src[] = {0, 1, 0};
    int dst[] = {1, 2, 2};

    p_dress_graph_t g = init_dress_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, DRESS_VARIANT_UNDIRECTED, 0);

    ASSERT(g != NULL, "init_dress_graph returned NULL");
    ASSERT_EQ_INT(g->N, 3, "N == 3");
    ASSERT_EQ_INT(g->E, 3, "E == 3");
    ASSERT_EQ_INT(g->variant, DRESS_VARIANT_UNDIRECTED, "variant == UNDIRECTED");

    /* Edge endpoints preserved */
    ASSERT_EQ_INT(g->U[0], 0, "U[0] == 0");
    ASSERT_EQ_INT(g->V[0], 1, "V[0] == 1");
    ASSERT_EQ_INT(g->U[1], 1, "U[1] == 1");
    ASSERT_EQ_INT(g->V[1], 2, "V[1] == 2");

    free_dress_graph(g);
}

static void test_weighted_triangle(void)
{
    printf("test_weighted_triangle\n");

    int    src[] = {0, 1, 0};
    int    dst[] = {1, 2, 2};
    double wts[] = {1.0, 2.0, 3.0};

    p_dress_graph_t g = init_dress_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        dup_double(wts, 3), DRESS_VARIANT_UNDIRECTED, 0);

    ASSERT(g != NULL, "init_dress_graph returned NULL");
    ASSERT_EQ_INT(g->E, 3, "E == 3");

    /* Weighted undirected edges get doubled weight (w(u,v)+w(v,u)) */
    ASSERT_NEAR(g->edge_weight[0], 2.0, 1e-12, "w[0] == 2.0");
    ASSERT_NEAR(g->edge_weight[1], 4.0, 1e-12, "w[1] == 4.0");
    ASSERT_NEAR(g->edge_weight[2], 6.0, 1e-12, "w[2] == 6.0");

    free_dress_graph(g);
}

static void test_all_variants(void)
{
    printf("test_all_variants\n");

    dress_variant_t variants[] = {
        DRESS_VARIANT_UNDIRECTED,
        DRESS_VARIANT_DIRECTED,
        DRESS_VARIANT_FORWARD,
        DRESS_VARIANT_BACKWARD
    };

    int src[] = {0, 1, 0};
    int dst[] = {1, 2, 2};

    for (int i = 0; i < 4; i++) {
        p_dress_graph_t g = init_dress_graph(
            3, 3, dup_int(src, 3), dup_int(dst, 3),
            NULL, variants[i], 0);
        ASSERT(g != NULL, "init_dress_graph succeeded for variant");
        ASSERT_EQ_INT(g->variant, variants[i], "variant stored correctly");
        ASSERT_EQ_INT(g->E, 3, "E == 3");
        free_dress_graph(g);
    }
}

static void test_precompute_intercepts(void)
{
    printf("test_precompute_intercepts\n");

    int src[] = {0, 1, 0};
    int dst[] = {1, 2, 2};

    p_dress_graph_t g = init_dress_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, DRESS_VARIANT_UNDIRECTED, 1);

    ASSERT(g != NULL, "init_dress_graph returned NULL");
    ASSERT_EQ_INT(g->precompute_intercepts, 1, "intercepts enabled");
    ASSERT(g->intercept_offset != NULL, "intercept_offset allocated");

    free_dress_graph(g);
}

/* ── test: fitting ─────────────────────────────────────────────────── */

static void test_triangle_convergence(void)
{
    printf("test_triangle_convergence\n");

    int src[] = {0, 1, 0};
    int dst[] = {1, 2, 2};

    p_dress_graph_t g = init_dress_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, DRESS_VARIANT_UNDIRECTED, 0);

    int    iters = 0;
    double delta = 0.0;
    fit(g, 100, 1e-8, &iters, &delta);

    ASSERT_GT(iters, 0, "iterations > 0");
    ASSERT_GT(delta, -1.0, "delta is non-negative (or nearly zero)");

    free_dress_graph(g);
}

static void test_triangle_equal_dress(void)
{
    printf("test_triangle_equal_dress\n");

    int src[] = {0, 1, 0};
    int dst[] = {1, 2, 2};

    p_dress_graph_t g = init_dress_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, DRESS_VARIANT_UNDIRECTED, 0);

    fit(g, 100, 1e-8, NULL, NULL);

    /* All edges in K3 are symmetric — dress values should be equal. */
    double d0 = g->edge_dress[0];
    ASSERT_NEAR(g->edge_dress[1], d0, 1e-6,
                "edge 1 dress == edge 0 dress");
    ASSERT_NEAR(g->edge_dress[2], d0, 1e-6,
                "edge 2 dress == edge 0 dress");

    free_dress_graph(g);
}

static void test_path_positive_dress(void)
{
    printf("test_path_positive_dress\n");

    /* Path graph: 0-1-2-3 */
    int src[] = {0, 1, 2};
    int dst[] = {1, 2, 3};

    p_dress_graph_t g = init_dress_graph(
        4, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, DRESS_VARIANT_UNDIRECTED, 0);

    fit(g, 100, 1e-6, NULL, NULL);

    for (int e = 0; e < 3; e++) {
        ASSERT_GT(g->edge_dress[e], 0.0, "path edge dress > 0 (self-loop term)");
        ASSERT_LT(g->edge_dress[e], 2.0, "path edge dress < 2");
    }

    free_dress_graph(g);
}

static void test_path_symmetry(void)
{
    printf("test_path_symmetry\n");

    int src[] = {0, 1, 2};
    int dst[] = {1, 2, 3};

    p_dress_graph_t g = init_dress_graph(
        4, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, DRESS_VARIANT_UNDIRECTED, 0);

    fit(g, 100, 1e-6, NULL, NULL);

    /* Endpoint edges (0-1 and 2-3) should match by symmetry. */
    ASSERT_NEAR(g->edge_dress[0], g->edge_dress[2], 1e-10,
                "endpoint edges have same dress");

    free_dress_graph(g);
}

static void test_fit_with_intercepts(void)
{
    printf("test_fit_with_intercepts\n");

    int src[] = {0, 1, 0};
    int dst[] = {1, 2, 2};

    /* Build graph with precomputed intercepts */
    p_dress_graph_t g1 = init_dress_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, DRESS_VARIANT_UNDIRECTED, 1);
    /* Build same graph without intercepts */
    p_dress_graph_t g2 = init_dress_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, DRESS_VARIANT_UNDIRECTED, 0);

    fit(g1, 100, 1e-10, NULL, NULL);
    fit(g2, 100, 1e-10, NULL, NULL);

    /* Results with and without intercepts should match. */
    for (int e = 0; e < 3; e++) {
        ASSERT_NEAR(g1->edge_dress[e], g2->edge_dress[e], 1e-8,
                    "intercept results match non-intercept");
    }

    free_dress_graph(g1);
    free_dress_graph(g2);
}

static void test_node_dress(void)
{
    printf("test_node_dress\n");

    int src[] = {0, 1, 0};
    int dst[] = {1, 2, 2};

    p_dress_graph_t g = init_dress_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, DRESS_VARIANT_UNDIRECTED, 0);

    fit(g, 100, 1e-8, NULL, NULL);

    /* All nodes in K3 have the same degree and structure → same node_dress. */
    ASSERT_GT(g->node_dress[0], 0.0, "node_dress[0] > 0");
    ASSERT_NEAR(g->node_dress[0], g->node_dress[1], 1e-6,
                "node_dress[0] == node_dress[1]");
    ASSERT_NEAR(g->node_dress[0], g->node_dress[2], 1e-6,
                "node_dress[0] == node_dress[2]");

    free_dress_graph(g);
}

static void test_weighted_fit(void)
{
    printf("test_weighted_fit\n");

    int    src[] = {0, 1, 0};
    int    dst[] = {1, 2, 2};
    double wts[] = {1.0, 2.0, 3.0};

    p_dress_graph_t g = init_dress_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        dup_double(wts, 3), DRESS_VARIANT_UNDIRECTED, 0);

    int iters = 0;
    fit(g, 100, 1e-6, &iters, NULL);
    ASSERT_GT(iters, 0, "weighted fit iterations > 0");

    /* With asymmetric weights edges should differ. */
    double d_min = g->edge_dress[0];
    double d_max = g->edge_dress[0];
    for (int e = 1; e < 3; e++) {
        if (g->edge_dress[e] < d_min) d_min = g->edge_dress[e];
        if (g->edge_dress[e] > d_max) d_max = g->edge_dress[e];
    }
    ASSERT_GT(d_max - d_min, 1e-6,
              "asymmetric weights produce different dress values");

    free_dress_graph(g);
}

static void test_single_edge(void)
{
    printf("test_single_edge\n");

    int src[] = {0};
    int dst[] = {1};

    p_dress_graph_t g = init_dress_graph(
        2, 1, dup_int(src, 1), dup_int(dst, 1),
        NULL, DRESS_VARIANT_UNDIRECTED, 0);

    ASSERT(g != NULL, "single edge graph created");
    ASSERT_EQ_INT(g->N, 2, "N == 2");
    ASSERT_EQ_INT(g->E, 1, "E == 1");

    fit(g, 100, 1e-8, NULL, NULL);

    /* Single edge with no common neighbors — dress comes only from the
       self-loop constant and should be small and positive. */
    ASSERT_GT(g->edge_dress[0], 0.0, "single edge dress > 0");

    free_dress_graph(g);
}

static void test_complete_graph_k4(void)
{
    printf("test_complete_graph_k4\n");

    /* K4 edge list: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) */
    int src[] = {0, 0, 0, 1, 1, 2};
    int dst[] = {1, 2, 3, 2, 3, 3};

    p_dress_graph_t g = init_dress_graph(
        4, 6, dup_int(src, 6), dup_int(dst, 6),
        NULL, DRESS_VARIANT_UNDIRECTED, 0);

    fit(g, 200, 1e-10, NULL, NULL);

    /* K4 is vertex-transitive: all edges should have the same dress. */
    double d0 = g->edge_dress[0];
    for (int e = 1; e < 6; e++) {
        ASSERT_NEAR(g->edge_dress[e], d0, 1e-6,
                    "K4 edges have equal dress");
    }

    /* All nodes should also have equal dress norm. */
    for (int u = 1; u < 4; u++) {
        ASSERT_NEAR(g->node_dress[u], g->node_dress[0], 1e-6,
                    "K4 nodes have equal dress norm");
    }

    free_dress_graph(g);
}

static void test_star_graph(void)
{
    printf("test_star_graph\n");

    /* Star with center 0 and leaves 1,2,3,4 */
    int src[] = {0, 0, 0, 0};
    int dst[] = {1, 2, 3, 4};

    p_dress_graph_t g = init_dress_graph(
        5, 4, dup_int(src, 4), dup_int(dst, 4),
        NULL, DRESS_VARIANT_UNDIRECTED, 0);

    fit(g, 100, 1e-8, NULL, NULL);

    /* All edges should have equal dress by symmetry. */
    double d0 = g->edge_dress[0];
    for (int e = 1; e < 4; e++) {
        ASSERT_NEAR(g->edge_dress[e], d0, 1e-6,
                    "star edges have equal dress");
    }

    free_dress_graph(g);
}

static void test_fit_null_out_params(void)
{
    printf("test_fit_null_out_params\n");

    int src[] = {0, 1};
    int dst[] = {1, 2};

    p_dress_graph_t g = init_dress_graph(
        3, 2, dup_int(src, 2), dup_int(dst, 2),
        NULL, DRESS_VARIANT_UNDIRECTED, 0);

    /* Should not crash when iters and delta are NULL. */
    fit(g, 10, 1e-6, NULL, NULL);

    ASSERT_GT(g->edge_dress[0], 0.0, "dress computed even with NULL out params");

    free_dress_graph(g);
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== DRESS C tests ===\n\n");

    /* construction */
    test_unweighted_triangle();
    test_weighted_triangle();
    test_all_variants();
    test_precompute_intercepts();

    /* fitting */
    test_triangle_convergence();
    test_triangle_equal_dress();
    test_path_positive_dress();
    test_path_symmetry();
    test_fit_with_intercepts();
    test_node_dress();
    test_weighted_fit();

    /* edge cases */
    test_single_edge();
    test_complete_graph_k4();
    test_star_graph();
    test_fit_null_out_params();

    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
