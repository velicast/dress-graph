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
#define ASSERT_EQ_DBL(a, b, msg)    ASSERT((a) == (b), msg)
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

    p_dress_graph_t g = dress_init_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    ASSERT(g != NULL, "dress_init_graph returned NULL");
    ASSERT_EQ_INT(g->N, 3, "N == 3");
    ASSERT_EQ_INT(g->E, 3, "E == 3");
    ASSERT_EQ_INT(g->variant, DRESS_VARIANT_UNDIRECTED, "variant == UNDIRECTED");

    /* Edge endpoints preserved */
    ASSERT_EQ_INT(g->U[0], 0, "U[0] == 0");
    ASSERT_EQ_INT(g->V[0], 1, "V[0] == 1");
    ASSERT_EQ_INT(g->U[1], 1, "U[1] == 1");
    ASSERT_EQ_INT(g->V[1], 2, "V[1] == 2");

    dress_free_graph(g);
}

static void test_weighted_triangle(void)
{
    printf("test_weighted_triangle\n");

    int    src[] = {0, 1, 0};
    int    dst[] = {1, 2, 2};
    double wts[] = {1.0, 2.0, 3.0};

    p_dress_graph_t g = dress_init_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        dup_double(wts, 3), NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    ASSERT(g != NULL, "dress_init_graph returned NULL");
    ASSERT_EQ_INT(g->E, 3, "E == 3");

    /* Weighted undirected edges get doubled weight (w(u,v)+w(v,u)) */
    ASSERT_NEAR(g->edge_weight[0], 2.0, 1e-12, "w[0] == 2.0");
    ASSERT_NEAR(g->edge_weight[1], 4.0, 1e-12, "w[1] == 4.0");
    ASSERT_NEAR(g->edge_weight[2], 6.0, 1e-12, "w[2] == 6.0");

    dress_free_graph(g);
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
        p_dress_graph_t g = dress_init_graph(
            3, 3, dup_int(src, 3), dup_int(dst, 3),
            NULL, NULL, variants[i], 0);
        ASSERT(g != NULL, "dress_init_graph succeeded for variant");
        ASSERT_EQ_INT(g->variant, variants[i], "variant stored correctly");
        ASSERT_EQ_INT(g->E, 3, "E == 3");
        dress_free_graph(g);
    }
}

static void test_precompute_intercepts(void)
{
    printf("test_precompute_intercepts\n");

    int src[] = {0, 1, 0};
    int dst[] = {1, 2, 2};

    p_dress_graph_t g = dress_init_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 1);

    ASSERT(g != NULL, "dress_init_graph returned NULL");
    ASSERT_EQ_INT(g->precompute_intercepts, 1, "intercepts enabled");
    ASSERT(g->intercept_offset != NULL, "intercept_offset allocated");

    dress_free_graph(g);
}

/* ── test: fitting ─────────────────────────────────────────────────── */

static void test_triangle_convergence(void)
{
    printf("test_triangle_convergence\n");

    int src[] = {0, 1, 0};
    int dst[] = {1, 2, 2};

    p_dress_graph_t g = dress_init_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    int    iters = 0;
    double delta = 0.0;
    dress_fit(g, 100, 1e-8, &iters, &delta);

    ASSERT_GT(iters, 0, "iterations > 0");
    ASSERT_GT(delta, -1.0, "delta is non-negative (or nearly zero)");

    dress_free_graph(g);
}

static void test_vertex_weights_default(void)
{
    printf("test_vertex_weights_default\n");

    int src[] = {0, 1, 0};
    int dst[] = {1, 2, 2};
    int N = 3, E = 3;

    // 1. Default (implicit All-1 vertex weights)
    p_dress_graph_t g1 = dress_init_graph(
        N, E, dup_int(src, E), dup_int(dst, E),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    dress_fit(g1, 100, 1e-8, NULL, NULL);

    // 2. Explicit All-1 vertex weights
    double nw[] = {1.0, 1.0, 1.0};
    double *nw_copy = (double *)malloc(N * sizeof(double));
    memcpy(nw_copy, nw, N * sizeof(double));

    p_dress_graph_t g2 = dress_init_graph(
        N, E, dup_int(src, E), dup_int(dst, E),
        NULL, nw_copy,
        DRESS_VARIANT_UNDIRECTED, 0);

    dress_fit(g2, 100, 1e-8, NULL, NULL);

    int i;
    for (i = 0; i < E; i++) {
        ASSERT_NEAR(g1->edge_dress[i], g2->edge_dress[i], 1e-12,
                    "default vs explicit vertex weights match (edge_dress)");
    }
    for (i = 0; i < N; i++) {
        ASSERT_NEAR(g1->vertex_dress[i], g2->vertex_dress[i], 1e-12,
                    "default vs explicit vertex weights match (vertex_dress)");
    }

    dress_free_graph(g1);
    dress_free_graph(g2);
}

static void test_triangle_equal_dress(void)
{
    printf("test_triangle_equal_dress\n");

    int src[] = {0, 1, 0};
    int dst[] = {1, 2, 2};

    p_dress_graph_t g = dress_init_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    dress_fit(g, 100, 1e-8, NULL, NULL);

    /* All edges in K3 are symmetric — dress values should be equal. */
    double d0 = g->edge_dress[0];
    ASSERT_NEAR(g->edge_dress[1], d0, 1e-6,
                "edge 1 dress == edge 0 dress");
    ASSERT_NEAR(g->edge_dress[2], d0, 1e-6,
                "edge 2 dress == edge 0 dress");

    dress_free_graph(g);
}

static void test_path_positive_dress(void)
{
    printf("test_path_positive_dress\n");

    /* Path graph: 0-1-2-3 */
    int src[] = {0, 1, 2};
    int dst[] = {1, 2, 3};

    p_dress_graph_t g = dress_init_graph(
        4, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    dress_fit(g, 100, 1e-6, NULL, NULL);

    for (int e = 0; e < 3; e++) {
        ASSERT_GT(g->edge_dress[e], 0.0, "path edge dress > 0 (self-loop term)");
        ASSERT_LT(g->edge_dress[e], 2.0, "path edge dress < 2");
    }

    dress_free_graph(g);
}

static void test_path_symmetry(void)
{
    printf("test_path_symmetry\n");

    int src[] = {0, 1, 2};
    int dst[] = {1, 2, 3};

    p_dress_graph_t g = dress_init_graph(
        4, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    dress_fit(g, 100, 1e-6, NULL, NULL);

    /* Endpoint edges (0-1 and 2-3) should match by symmetry. */
    ASSERT_NEAR(g->edge_dress[0], g->edge_dress[2], 1e-10,
                "endpoint edges have same dress");

    dress_free_graph(g);
}

static void test_fit_with_intercepts(void)
{
    printf("test_fit_with_intercepts\n");

    int src[] = {0, 1, 0};
    int dst[] = {1, 2, 2};

    /* Build graph with precomputed intercepts */
    p_dress_graph_t g1 = dress_init_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 1);
    /* Build same graph without intercepts */
    p_dress_graph_t g2 = dress_init_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    dress_fit(g1, 100, 1e-10, NULL, NULL);
    dress_fit(g2, 100, 1e-10, NULL, NULL);

    /* Results with and without intercepts should match. */
    for (int e = 0; e < 3; e++) {
        ASSERT_NEAR(g1->edge_dress[e], g2->edge_dress[e], 1e-8,
                    "intercept results match non-intercept");
    }

    dress_free_graph(g1);
    dress_free_graph(g2);
}

static void test_vertex_dress(void)
{
    printf("test_vertex_dress\n");

    int src[] = {0, 1, 0};
    int dst[] = {1, 2, 2};

    p_dress_graph_t g = dress_init_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    dress_fit(g, 100, 1e-8, NULL, NULL);

    /* All nodes in K3 have the same degree and structure → same vertex_dress. */
    ASSERT_GT(g->vertex_dress[0], 0.0, "vertex_dress[0] > 0");
    ASSERT_NEAR(g->vertex_dress[0], g->vertex_dress[1], 1e-6,
                "vertex_dress[0] == vertex_dress[1]");
    ASSERT_NEAR(g->vertex_dress[0], g->vertex_dress[2], 1e-6,
                "vertex_dress[0] == vertex_dress[2]");

    dress_free_graph(g);
}

static void test_weighted_fit(void)
{
    printf("test_weighted_fit\n");

    int    src[] = {0, 1, 0};
    int    dst[] = {1, 2, 2};
    double wts[] = {1.0, 2.0, 3.0};

    p_dress_graph_t g = dress_init_graph(
        3, 3, dup_int(src, 3), dup_int(dst, 3),
        dup_double(wts, 3), NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    int iters = 0;
    dress_fit(g, 100, 1e-6, &iters, NULL);
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

    dress_free_graph(g);
}

static void test_single_edge(void)
{
    printf("test_single_edge\n");

    int src[] = {0};
    int dst[] = {1};

    p_dress_graph_t g = dress_init_graph(
        2, 1, dup_int(src, 1), dup_int(dst, 1),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    ASSERT(g != NULL, "single edge graph created");
    ASSERT_EQ_INT(g->N, 2, "N == 2");
    ASSERT_EQ_INT(g->E, 1, "E == 1");

    dress_fit(g, 100, 1e-8, NULL, NULL);

    /* Single edge with no common neighbors — dress comes only from the
       self-loop constant and should be small and positive. */
    ASSERT_GT(g->edge_dress[0], 0.0, "single edge dress > 0");

    dress_free_graph(g);
}

static void test_complete_graph_k4(void)
{
    printf("test_complete_graph_k4\n");

    /* K4 edge list: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) */
    int src[] = {0, 0, 0, 1, 1, 2};
    int dst[] = {1, 2, 3, 2, 3, 3};

    p_dress_graph_t g = dress_init_graph(
        4, 6, dup_int(src, 6), dup_int(dst, 6),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    dress_fit(g, 200, 1e-10, NULL, NULL);

    /* K4 is vertex-transitive: all edges should have the same dress. */
    double d0 = g->edge_dress[0];
    for (int e = 1; e < 6; e++) {
        ASSERT_NEAR(g->edge_dress[e], d0, 1e-6,
                    "K4 edges have equal dress");
    }

    /* All nodes should also have equal dress norm. */
    for (int u = 1; u < 4; u++) {
        ASSERT_NEAR(g->vertex_dress[u], g->vertex_dress[0], 1e-6,
                    "K4 nodes have equal dress norm");
    }

    dress_free_graph(g);
}

static void test_star_graph(void)
{
    printf("test_star_graph\n");

    /* Star with center 0 and leaves 1,2,3,4 */
    int src[] = {0, 0, 0, 0};
    int dst[] = {1, 2, 3, 4};

    p_dress_graph_t g = dress_init_graph(
        5, 4, dup_int(src, 4), dup_int(dst, 4),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    dress_fit(g, 100, 1e-8, NULL, NULL);

    /* All edges should have equal dress by symmetry. */
    double d0 = g->edge_dress[0];
    for (int e = 1; e < 4; e++) {
        ASSERT_NEAR(g->edge_dress[e], d0, 1e-6,
                    "star edges have equal dress");
    }

    dress_free_graph(g);
}

static void test_fit_null_out_params(void)
{
    printf("test_fit_null_out_params\n");

    int src[] = {0, 1};
    int dst[] = {1, 2};

    p_dress_graph_t g = dress_init_graph(
        3, 2, dup_int(src, 2), dup_int(dst, 2),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    /* Should not crash when iters and delta are NULL. */
    dress_fit(g, 10, 1e-6, NULL, NULL);

    ASSERT_GT(g->edge_dress[0], 0.0, "dress computed even with NULL out params");

    dress_free_graph(g);
}

/* ── test: label-independence (sort+KBN) ───────────────────────────── */

/*
 * Apply a vertex permutation to a graph's edge list.
 *   perm[old_id] = new_id
 * The caller must free the returned arrays when done.
 */
static void permute_edges(const int *src, const int *dst, int E,
                          const int *perm,
                          int **out_src, int **out_dst)
{
    int *ns = (int *)malloc((size_t)E * sizeof(int));
    int *nd = (int *)malloc((size_t)E * sizeof(int));
    for (int i = 0; i < E; i++) {
        ns[i] = perm[src[i]];
        nd[i] = perm[dst[i]];
    }
    *out_src = ns;
    *out_dst = nd;
}

/* Compare qsort callback for doubles (ascending). */
static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/*
 * Build the sorted fingerprint: sort a copy of the dress arrays,
 * then memcmp.  This is exactly the product we sell.
 */
static void assert_fingerprint_equal(p_dress_graph_t g1, p_dress_graph_t g2,
                                     const char *label)
{
    /* ---- edge fingerprint ---- */
    int E = g1->E;
    ASSERT_EQ_INT(E, g2->E, "edge count mismatch");

    double *e1 = (double *)malloc((size_t)E * sizeof(double));
    double *e2 = (double *)malloc((size_t)E * sizeof(double));
    memcpy(e1, g1->edge_dress, (size_t)E * sizeof(double));
    memcpy(e2, g2->edge_dress, (size_t)E * sizeof(double));
    qsort(e1, (size_t)E, sizeof(double), cmp_double);
    qsort(e2, (size_t)E, sizeof(double), cmp_double);
    ASSERT(memcmp(e1, e2, (size_t)E * sizeof(double)) == 0, label);
    free(e1);
    free(e2);

    /* ---- vertex fingerprint ---- */
    int N = g1->N;
    ASSERT_EQ_INT(N, g2->N, "vertex count mismatch");

    double *n1 = (double *)malloc((size_t)N * sizeof(double));
    double *n2 = (double *)malloc((size_t)N * sizeof(double));
    memcpy(n1, g1->vertex_dress, (size_t)N * sizeof(double));
    memcpy(n2, g2->vertex_dress, (size_t)N * sizeof(double));
    qsort(n1, (size_t)N, sizeof(double), cmp_double);
    qsort(n2, (size_t)N, sizeof(double), cmp_double);
    ASSERT(memcmp(n1, n2, (size_t)N * sizeof(double)) == 0, label);
    free(n1);
    free(n2);
}

static void test_relabel_petersen(void)
{
    printf("test_relabel_petersen\n");

    /*
     * Petersen graph (10 vertices, 15 edges).
     * Non-trivial, 3-regular, not vertex-transitive in trivial way.
     */
    int src[] = {0,0,0, 1,1, 2,2, 3,3, 4,4, 5,6,7,8};
    int dst[] = {1,4,5, 2,6, 3,7, 4,8, 9, 7,8,9,9,5};
    int N = 10, E = 15;

    p_dress_graph_t g1 = dress_init_graph(
        N, E, dup_int(src, E), dup_int(dst, E),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);
    dress_fit(g1, 200, 1e-12, NULL, NULL);

    /* Random-looking permutation */
    int perm[] = {7, 2, 5, 0, 9, 1, 8, 4, 3, 6};
    int *ps, *pd;
    permute_edges(src, dst, E, perm, &ps, &pd);

    p_dress_graph_t g2 = dress_init_graph(
        N, E, ps, pd,
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);
    dress_fit(g2, 200, 1e-12, NULL, NULL);

    assert_fingerprint_equal(g1, g2,
        "Petersen relabeled: sorted fingerprint must be bitwise identical");

    dress_free_graph(g1);
    dress_free_graph(g2);
}

static void test_relabel_weighted(void)
{
    printf("test_relabel_weighted\n");

    /* Weighted house graph (5 verts, 6 edges) */
    int    src[] = {0, 0, 1, 2, 2, 3};
    int    dst[] = {1, 3, 2, 3, 4, 4};
    double wts[] = {1.0, 3.0, 2.0, 5.0, 4.0, 7.0};
    int N = 5, E = 6;

    p_dress_graph_t g1 = dress_init_graph(
        N, E, dup_int(src, E), dup_int(dst, E),
        dup_double(wts, E), NULL,
        DRESS_VARIANT_UNDIRECTED, 0);
    dress_fit(g1, 200, 1e-12, NULL, NULL);

    int perm[] = {3, 0, 4, 1, 2};
    int *ps, *pd;
    permute_edges(src, dst, E, perm, &ps, &pd);

    p_dress_graph_t g2 = dress_init_graph(
        N, E, ps, pd,
        dup_double(wts, E), NULL,
        DRESS_VARIANT_UNDIRECTED, 0);
    dress_fit(g2, 200, 1e-12, NULL, NULL);

    assert_fingerprint_equal(g1, g2,
        "weighted relabeled: sorted fingerprint must be bitwise identical");

    dress_free_graph(g1);
    dress_free_graph(g2);
}

static void test_edge_reorder(void)
{
    printf("test_edge_reorder\n");

    /* Same graph, edges listed in different order.
     * Path: 0-1-2-3-4 with a triangle 1-2-3. */
    int src1[] = {0, 1, 2, 3, 1};
    int dst1[] = {1, 2, 3, 4, 3};
    int src2[] = {1, 3, 2, 1, 0};
    int dst2[] = {3, 4, 3, 2, 1};
    int N = 5, E = 5;

    p_dress_graph_t g1 = dress_init_graph(
        N, E, dup_int(src1, E), dup_int(dst1, E),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);
    p_dress_graph_t g2 = dress_init_graph(
        N, E, dup_int(src2, E), dup_int(dst2, E),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

    dress_fit(g1, 200, 1e-12, NULL, NULL);
    dress_fit(g2, 200, 1e-12, NULL, NULL);

    assert_fingerprint_equal(g1, g2,
        "edge-reordered: sorted fingerprint must be bitwise identical");

    dress_free_graph(g1);
    dress_free_graph(g2);
}

static void test_relabel_with_intercepts(void)
{
    printf("test_relabel_with_intercepts\n");

    /* Petersen graph with precomputed intercepts. */
    int src[] = {0,0,0, 1,1, 2,2, 3,3, 4,4, 5,6,7,8};
    int dst[] = {1,4,5, 2,6, 3,7, 4,8, 9, 7,8,9,9,5};
    int N = 10, E = 15;

    p_dress_graph_t g1 = dress_init_graph(
        N, E, dup_int(src, E), dup_int(dst, E),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 1);
    dress_fit(g1, 200, 1e-12, NULL, NULL);

    int perm[] = {7, 2, 5, 0, 9, 1, 8, 4, 3, 6};
    int *ps, *pd;
    permute_edges(src, dst, E, perm, &ps, &pd);

    p_dress_graph_t g2 = dress_init_graph(
        N, E, ps, pd,
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 1);
    dress_fit(g2, 200, 1e-12, NULL, NULL);

    assert_fingerprint_equal(g1, g2,
        "intercept relabeled: sorted fingerprint must be bitwise identical");

    dress_free_graph(g1);
    dress_free_graph(g2);
}

static void test_relabel_directed(void)
{
    printf("test_relabel_directed\n");

    /* Directed triangle: 0→1, 1→2, 2→0 */
    int src[] = {0, 1, 2};
    int dst[] = {1, 2, 0};
    int N = 3, E = 3;
    const char *names[] = {"DIRECTED", "FORWARD", "BACKWARD"};
    dress_variant_t dvars[] = {DRESS_VARIANT_DIRECTED,
                               DRESS_VARIANT_FORWARD,
                               DRESS_VARIANT_BACKWARD};

    for (int vi = 0; vi < 3; vi++) {
        p_dress_graph_t g1 = dress_init_graph(
            N, E, dup_int(src, E), dup_int(dst, E),
            NULL, NULL, dvars[vi], 0);
        dress_fit(g1, 200, 1e-12, NULL, NULL);

        int perm[] = {2, 0, 1};
        int *ps, *pd;
        permute_edges(src, dst, E, perm, &ps, &pd);

        p_dress_graph_t g2 = dress_init_graph(
            N, E, ps, pd,
            NULL, NULL, dvars[vi], 0);
        dress_fit(g2, 200, 1e-12, NULL, NULL);

        char msg[128];
        snprintf(msg, sizeof(msg),
                 "%s relabeled: sorted fingerprint must be bitwise identical",
                 names[vi]);
        assert_fingerprint_equal(g1, g2, msg);

        dress_free_graph(g1);
        dress_free_graph(g2);
    }
}

/* ── test: dress_get ───────────────────────────────────────────────── */

static void test_dress_get_existing_edge(void)
{
    printf("test_dress_get_existing_edge\n");

    /* K4 graph */
    int src[] = {0, 0, 0, 1, 1, 2};
    int dst[] = {1, 2, 3, 2, 3, 3};
    int N = 4, E = 6;

    p_dress_graph_t g = dress_init_graph(
        N, E, dup_int(src, E), dup_int(dst, E),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);
    dress_fit(g, 200, 1e-12, NULL, NULL);

    for (int e = 0; e < E; e++) {
        double expected = g->edge_dress[e];
        double got_uv = dress_get(g, src[e], dst[e], 100, 1e-12, 1.0);
        double got_vu = dress_get(g, dst[e], src[e], 100, 1e-12, 1.0);
        ASSERT_EQ_DBL(got_uv, expected,
                      "dress_get(u,v) == edge_dress for existing edge");
        ASSERT_EQ_DBL(got_vu, expected,
                      "dress_get(v,u) == edge_dress for existing edge");
    }

    dress_free_graph(g);
}

static void test_dress_get_virtual_edge(void)
{
    printf("test_dress_get_virtual_edge\n");

    /* Path 0-1-2-3: vertices 0 and 3 have no direct edge. */
    int src[] = {0, 1, 2};
    int dst[] = {1, 2, 3};

    p_dress_graph_t g = dress_init_graph(
        4, 3, dup_int(src, 3), dup_int(dst, 3),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);
    dress_fit(g, 200, 1e-12, NULL, NULL);

    double d03 = dress_get(g, 0, 3, 200, 1e-12, 1.0);
    double d30 = dress_get(g, 3, 0, 200, 1e-12, 1.0);
    ASSERT_GT(d03, 0.0, "virtual edge dress > 0");
    ASSERT_LT(d03, 2.0, "virtual edge dress < 2");
    ASSERT_NEAR(d03, d30, 1e-12, "virtual edge symmetric");

    /* Virtual edge (0,2): one common neighbor → higher dress. */
    double d02 = dress_get(g, 0, 2, 200, 1e-12, 1.0);
    ASSERT_GT(d02, d03, "virtual edge with common neighbor > without");

    dress_free_graph(g);
}

static void test_virtual_edge_relabel_invariance(void)
{
    printf("test_virtual_edge_relabel_invariance\n");

    /* Cycle C5: 0-1-2-3-4-0.  All virtual edges should form the
     * same sorted fingerprint under relabeling. */
    int src[] = {0, 1, 2, 3, 4};
    int dst[] = {1, 2, 3, 4, 0};
    int N = 5, E = 5;

    p_dress_graph_t g1 = dress_init_graph(
        N, E, dup_int(src, E), dup_int(dst, E),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);
    dress_fit(g1, 200, 1e-12, NULL, NULL);

    int perm[] = {4, 3, 2, 1, 0};
    int *ps, *pd;
    permute_edges(src, dst, E, perm, &ps, &pd);

    p_dress_graph_t g2 = dress_init_graph(
        N, E, ps, pd,
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);
    dress_fit(g2, 200, 1e-12, NULL, NULL);

    /* Collect all virtual-edge dress values for both graphs, sort, memcmp. */
    int n_virtual = N * (N - 1) / 2 - E;  /* C5: 10 - 5 = 5 virtual edges */
    double *v1 = (double *)malloc((size_t)n_virtual * sizeof(double));
    double *v2 = (double *)malloc((size_t)n_virtual * sizeof(double));
    int k = 0;
    for (int u = 0; u < N; u++) {
        for (int v = u + 1; v < N; v++) {
            /* Skip existing edges: in C5, adjacent vertices differ by 1 (mod 5). */
            int diff = v - u;
            if (diff == 1 || diff == N - 1) continue;
            v1[k] = dress_get(g1, u, v, 200, 1e-12, 1.0);
            v2[k] = dress_get(g2, u, v, 200, 1e-12, 1.0);
            k++;
        }
    }
    ASSERT_EQ_INT(k, n_virtual, "virtual edge count");
    qsort(v1, (size_t)k, sizeof(double), cmp_double);
    qsort(v2, (size_t)k, sizeof(double), cmp_double);
    ASSERT(memcmp(v1, v2, (size_t)k * sizeof(double)) == 0,
           "virtual edge sorted fingerprint must be bitwise identical");
    free(v1);
    free(v2);

    dress_free_graph(g1);
    dress_free_graph(g2);
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
    test_vertex_weights_default();
    test_triangle_equal_dress();
    test_path_positive_dress();
    test_path_symmetry();
    test_fit_with_intercepts();
    test_vertex_dress();
    test_weighted_fit();

    /* edge cases */
    test_single_edge();
    test_complete_graph_k4();
    test_star_graph();
    test_fit_null_out_params();

    /* label-independence: sort + memcmp (the product) */
    test_relabel_petersen();
    test_relabel_weighted();
    test_edge_reorder();
    test_relabel_with_intercepts();
    test_relabel_directed();

    /* dress_get */
    test_dress_get_existing_edge();
    test_dress_get_virtual_edge();
    test_virtual_edge_relabel_invariance();

    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
