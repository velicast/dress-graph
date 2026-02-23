/*
 * Tests for the DRESS igraph wrapper (libdress-igraph).
 *
 * Build from the repo root:
 *   gcc -O2 -I libdress/include -I libdress-igraph/include \
 *       $(pkg-config --cflags igraph) \
 *       -o tests/c/test_dress_igraph \
 *       tests/c/test_dress_igraph.c \
 *       libdress-igraph/src/dress_igraph.c \
 *       libdress/src/dress.c \
 *       $(pkg-config --libs igraph) -lm -fopenmp
 *
 * Run:
 *   ./tests/c/test_dress_igraph
 */

#include "dress_igraph.h"
#include "dress/dress.h"

#include <igraph/igraph.h>
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

/* ── helper: build igraph from edge arrays ─────────────────────────── */

static igraph_t make_graph(const int *edges, int n_edges, int n_vertices,
                           igraph_bool_t directed)
{
    igraph_t g;
    igraph_vector_int_t ev;
    igraph_vector_int_init(&ev, 2 * n_edges);
    for (int i = 0; i < n_edges; i++) {
        VECTOR(ev)[2 * i]     = edges[2 * i];
        VECTOR(ev)[2 * i + 1] = edges[2 * i + 1];
    }
    igraph_create(&g, &ev, n_vertices, directed);
    igraph_vector_int_destroy(&ev);
    return g;
}

/* ── test: unweighted triangle ─────────────────────────────────────── */

static void test_unweighted_triangle(void)
{
    printf("test_unweighted_triangle\n");

    int edges[] = {0,1, 1,2, 0,2};
    igraph_t g = make_graph(edges, 3, 3, IGRAPH_UNDIRECTED);

    dress_igraph_result_t result;
    int rc = dress_igraph_compute(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                                  100, 1e-8, 0, &result);

    ASSERT_EQ_INT(rc, 0, "compute returned 0");
    ASSERT_EQ_INT(result.N, 3, "N == 3");
    ASSERT_EQ_INT(result.E, 3, "E == 3");
    ASSERT_GT(result.iterations, 0, "iterations > 0");

    /* All edges in a triangle should have equal dress = 2.0 */
    for (int e = 0; e < result.E; e++) {
        ASSERT_NEAR(result.dress[e], 2.0, 1e-6,
                    "triangle edge dress == 2.0");
    }

    dress_igraph_free(&result);
    igraph_destroy(&g);
}

/* ── test: weighted triangle ───────────────────────────────────────── */

static void test_weighted_triangle(void)
{
    printf("test_weighted_triangle\n");

    int edges[] = {0,1, 1,2, 0,2};
    igraph_t g = make_graph(edges, 3, 3, IGRAPH_UNDIRECTED);

    /* Set edge weights via igraph attribute */
    double weights[] = {1.0, 2.0, 3.0};
    for (int e = 0; e < 3; e++) {
        SETEAN(&g, "weight", e, weights[e]);
    }

    dress_igraph_result_t result;
    int rc = dress_igraph_compute(&g, "weight", DRESS_VARIANT_UNDIRECTED,
                                  100, 1e-8, 0, &result);

    ASSERT_EQ_INT(rc, 0, "compute returned 0");
    ASSERT_EQ_INT(result.E, 3, "E == 3");

    /* With weights, dress values should still be > 0 and reasonable */
    for (int e = 0; e < result.E; e++) {
        ASSERT_GT(result.dress[e], 0.0, "weighted dress > 0");
        ASSERT_LT(result.dress[e], 3.0, "weighted dress < 3");
    }

    dress_igraph_free(&result);
    igraph_destroy(&g);
}

/* ── test: path graph ──────────────────────────────────────────────── */

static void test_path(void)
{
    printf("test_path\n");

    int edges[] = {0,1, 1,2, 2,3};
    igraph_t g = make_graph(edges, 3, 4, IGRAPH_UNDIRECTED);

    dress_igraph_result_t result;
    int rc = dress_igraph_compute(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                                  100, 1e-8, 0, &result);

    ASSERT_EQ_INT(rc, 0, "compute returned 0");
    ASSERT_EQ_INT(result.N, 4, "N == 4");
    ASSERT_EQ_INT(result.E, 3, "E == 3");

    /* All dress values positive and bounded */
    for (int e = 0; e < result.E; e++) {
        ASSERT_GT(result.dress[e], 0.0, "path dress > 0");
        ASSERT_LT(result.dress[e], 2.0 + 1e-9, "path dress <= 2");
    }

    /* Endpoint edges (0-1, 2-3) should be symmetric */
    ASSERT_NEAR(result.dress[0], result.dress[2], 1e-10,
                "endpoint edges symmetric");

    dress_igraph_free(&result);
    igraph_destroy(&g);
}

/* ── test: K4 complete graph ───────────────────────────────────────── */

static void test_k4(void)
{
    printf("test_k4\n");

    int edges[] = {0,1, 0,2, 0,3, 1,2, 1,3, 2,3};
    igraph_t g = make_graph(edges, 6, 4, IGRAPH_UNDIRECTED);

    dress_igraph_result_t result;
    int rc = dress_igraph_compute(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                                  100, 1e-8, 0, &result);

    ASSERT_EQ_INT(rc, 0, "compute returned 0");
    ASSERT_EQ_INT(result.E, 6, "K4 has 6 edges");

    /* All edges in K4 should have equal dress = 2.0 */
    double d0 = result.dress[0];
    for (int e = 1; e < result.E; e++) {
        ASSERT_NEAR(result.dress[e], d0, 1e-6, "K4 edges equal");
    }
    ASSERT_NEAR(d0, 2.0, 1e-6, "K4 dress == 2.0");

    dress_igraph_free(&result);
    igraph_destroy(&g);
}

/* ── test: precompute intercepts ───────────────────────────────────── */

static void test_precompute_intercepts(void)
{
    printf("test_precompute_intercepts\n");

    int edges[] = {0,1, 1,2, 0,2};
    igraph_t g = make_graph(edges, 3, 3, IGRAPH_UNDIRECTED);

    dress_igraph_result_t r_no, r_yes;

    dress_igraph_compute(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                         100, 1e-10, 0, &r_no);
    dress_igraph_compute(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                         100, 1e-10, 1, &r_yes);

    /* Both should give the same results */
    ASSERT_EQ_INT(r_no.E, r_yes.E, "same edge count");
    for (int e = 0; e < r_no.E; e++) {
        ASSERT_NEAR(r_no.dress[e], r_yes.dress[e], 1e-12,
                    "dress identical with/without intercepts");
    }

    dress_igraph_free(&r_no);
    dress_igraph_free(&r_yes);
    igraph_destroy(&g);
}

/* ── test: directed variants ───────────────────────────────────────── */

static void test_directed_variants(void)
{
    printf("test_directed_variants\n");

    /* Directed triangle: 0->1, 1->2, 0->2 */
    int edges[] = {0,1, 1,2, 0,2};
    igraph_t g = make_graph(edges, 3, 3, IGRAPH_DIRECTED);

    dress_variant_t variants[] = {
        DRESS_VARIANT_DIRECTED,
        DRESS_VARIANT_FORWARD,
        DRESS_VARIANT_BACKWARD
    };

    for (int v = 0; v < 3; v++) {
        dress_igraph_result_t result;
        int rc = dress_igraph_compute(&g, NULL, variants[v],
                                      100, 1e-8, 0, &result);
        ASSERT_EQ_INT(rc, 0, "directed compute returned 0");
        ASSERT_EQ_INT(result.E, 3, "directed E == 3");

        for (int e = 0; e < result.E; e++) {
            ASSERT_GT(result.dress[e], 0.0, "directed dress > 0");
        }

        dress_igraph_free(&result);
    }

    igraph_destroy(&g);
}

/* ── test: node dress values ───────────────────────────────────────── */

static void test_node_dress(void)
{
    printf("test_node_dress\n");

    int edges[] = {0,1, 1,2, 0,2};
    igraph_t g = make_graph(edges, 3, 3, IGRAPH_UNDIRECTED);

    dress_igraph_result_t result;
    dress_igraph_compute(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                         100, 1e-8, 0, &result);

    /* All node norms in a triangle should be equal and >= 2 */
    ASSERT(result.node_dress != NULL, "node_dress not NULL");
    double n0 = result.node_dress[0];
    ASSERT_GT(n0, 2.0 - 1e-6, "node dress >= 2");
    for (int u = 1; u < result.N; u++) {
        ASSERT_NEAR(result.node_dress[u], n0, 1e-6,
                    "triangle node norms equal");
    }

    dress_igraph_free(&result);
    igraph_destroy(&g);
}

/* ── test: dress_igraph_to_vector ──────────────────────────────────── */

static void test_to_vector(void)
{
    printf("test_to_vector\n");

    int edges[] = {0,1, 1,2, 0,2};
    igraph_t g = make_graph(edges, 3, 3, IGRAPH_UNDIRECTED);

    dress_igraph_result_t result;
    dress_igraph_compute(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                         100, 1e-8, 0, &result);

    igraph_vector_t vec;
    igraph_vector_init(&vec, 0);
    int rc = dress_igraph_to_vector(&result, &vec);

    ASSERT_EQ_INT(rc, 0, "to_vector returned 0");
    ASSERT_EQ_INT((int)igraph_vector_size(&vec), result.E,
                  "vector size == E");

    for (int e = 0; e < result.E; e++) {
        ASSERT_NEAR(VECTOR(vec)[e], result.dress[e], 1e-15,
                    "vector matches dress array");
    }

    igraph_vector_destroy(&vec);
    dress_igraph_free(&result);
    igraph_destroy(&g);
}

/* ── test: empty graph ─────────────────────────────────────────────── */

static void test_empty_graph(void)
{
    printf("test_empty_graph\n");

    igraph_t g;
    igraph_empty(&g, 5, IGRAPH_UNDIRECTED);

    dress_igraph_result_t result;
    int rc = dress_igraph_compute(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                                  100, 1e-8, 0, &result);

    ASSERT_EQ_INT(rc, 0, "compute on empty graph returned 0");
    ASSERT_EQ_INT(result.N, 5, "N == 5");
    ASSERT_EQ_INT(result.E, 0, "E == 0");

    dress_igraph_free(&result);
    igraph_destroy(&g);
}

/* ── test: null params ─────────────────────────────────────────────── */

static void test_null_params(void)
{
    printf("test_null_params\n");

    dress_igraph_result_t result;
    int rc;

    rc = dress_igraph_compute(NULL, NULL, DRESS_VARIANT_UNDIRECTED,
                              100, 1e-8, 0, &result);
    ASSERT(rc != 0, "NULL graph returns error");

    igraph_t g;
    igraph_empty(&g, 3, IGRAPH_UNDIRECTED);

    rc = dress_igraph_compute(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                              100, 1e-8, 0, NULL);
    ASSERT(rc != 0, "NULL result returns error");

    igraph_destroy(&g);
}

/* ── test: cross-validate with raw C API ───────────────────────────── */

static int *dup_int_arr(const int *src, int n)
{
    int *p = (int *)malloc((size_t)n * sizeof(int));
    memcpy(p, src, (size_t)n * sizeof(int));
    return p;
}

static void test_cross_validate_with_c_api(void)
{
    printf("test_cross_validate_with_c_api\n");

    /* Build via igraph */
    int edges[] = {0,1, 1,2, 2,3, 0,2};
    igraph_t g = make_graph(edges, 4, 4, IGRAPH_UNDIRECTED);

    dress_igraph_result_t ig_result;
    dress_igraph_compute(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                         100, 1e-12, 0, &ig_result);

    /* Build via raw C API with the same edge list */
    int src[] = {0, 1, 2, 0};
    int dst[] = {1, 2, 3, 2};
    p_dress_graph_t dg = init_dress_graph(
        4, 4, dup_int_arr(src, 4), dup_int_arr(dst, 4),
        NULL, DRESS_VARIANT_UNDIRECTED, 0);

    int iterations = 0;
    double delta = 0.0;
    fit(dg, 100, 1e-12, &iterations, &delta);

    /* Compare dress values (same edge order) */
    ASSERT_EQ_INT(ig_result.E, dg->E, "same edge count");
    for (int e = 0; e < dg->E; e++) {
        ASSERT_NEAR(ig_result.dress[e], dg->edge_dress[e], 1e-14,
                    "igraph matches raw C dress");
    }

    /* Compare node norms */
    for (int u = 0; u < dg->N; u++) {
        ASSERT_NEAR(ig_result.node_dress[u], dg->node_dress[u], 1e-14,
                    "igraph matches raw C node dress");
    }

    free_dress_graph(dg);
    dress_igraph_free(&ig_result);
    igraph_destroy(&g);
}

/* ── test: star graph ──────────────────────────────────────────────── */

static void test_star_graph(void)
{
    printf("test_star_graph\n");

    igraph_t g;
    igraph_star(&g, 5, IGRAPH_STAR_UNDIRECTED, 0);

    dress_igraph_result_t result;
    int rc = dress_igraph_compute(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                                  100, 1e-8, 0, &result);

    ASSERT_EQ_INT(rc, 0, "star compute ok");
    ASSERT_EQ_INT(result.E, 4, "star has 4 edges");

    /* All star edges should be equal by symmetry */
    double d0 = result.dress[0];
    for (int e = 1; e < result.E; e++) {
        ASSERT_NEAR(result.dress[e], d0, 1e-6, "star edges equal");
    }

    dress_igraph_free(&result);
    igraph_destroy(&g);
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(void)
{
    /* Enable igraph attribute handling */
    igraph_set_attribute_table(&igraph_cattribute_table);

    printf("=== DRESS igraph tests ===\n\n");

    test_unweighted_triangle();
    test_weighted_triangle();
    test_path();
    test_k4();
    test_precompute_intercepts();
    test_directed_variants();
    test_node_dress();
    test_to_vector();
    test_empty_graph();
    test_null_params();
    test_cross_validate_with_c_api();
    test_star_graph();

    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
