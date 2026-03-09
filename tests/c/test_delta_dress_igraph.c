/*
 * Tests for the Δ^k-DRESS igraph wrapper (libdress-igraph).
 *
 * Build from the repo root:
 *   gcc -O2 -I libdress/include -I libdress-igraph/include \
 *       $(pkg-config --cflags igraph) \
 *       -o tests/c/test_delta_dress_igraph \
 *       tests/c/test_delta_dress_igraph.c \
 *       libdress-igraph/src/dress_igraph.c \
 *       libdress/src/dress.c \
 *       libdress/src/delta_dress.c \
 *       $(pkg-config --libs igraph) -lm -fopenmp
 *
 * Run:
 *   ./tests/c/test_delta_dress_igraph
 */

#include <dress/igraph/dress.h>
#include <dress/dress.h>

/* Undo convenience macros — this test cross-validates against the core
   C API, so we need the un-redirected delta_dress_fit() symbol. */
#undef delta_dress_fit


#include <igraph/igraph.h>
#include <math.h>
#include <stdint.h>
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

/* ── test: Δ^0 on a triangle (all bins should sum to 3 edges) ────── */

static void test_delta0_triangle(void)
{
    printf("test_delta0_triangle\n");

    int edges[] = {0,1, 1,2, 0,2};
    igraph_t g = make_graph(edges, 3, 3, IGRAPH_UNDIRECTED);

    delta_dress_result_igraph_t result;
    int rc = delta_dress_fit_igraph(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                                        0, 100, 1e-6, 0, &result);

    ASSERT_EQ_INT(rc, 0, "delta compute returned 0");
    ASSERT_GT(result.hist_size, 0, "hist_size > 0");
    ASSERT(result.histogram != NULL, "histogram not NULL");

    /* Sum should equal 3 (the number of edges in a triangle) */
    int64_t total = 0;
    for (int i = 0; i < result.hist_size; i++)
        total += result.histogram[i];
    ASSERT_EQ_INT((int)total, 3, "Δ^0 triangle: sum == 3 edges");

    /* All dress values should be 2.0 → top bin */
    ASSERT_GT(result.histogram[result.hist_size - 1], 0,
              "top bin non-zero for triangle");

    delta_dress_free_igraph(&result);
    igraph_destroy(&g);
}

/* ── test: Δ^1 on a triangle ─────────────────────────────────────── */

static void test_delta1_triangle(void)
{
    printf("test_delta1_triangle\n");

    int edges[] = {0,1, 1,2, 0,2};
    igraph_t g = make_graph(edges, 3, 3, IGRAPH_UNDIRECTED);

    delta_dress_result_igraph_t result;
    int rc = delta_dress_fit_igraph(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                                        1, 100, 1e-6, 0, &result);

    ASSERT_EQ_INT(rc, 0, "delta compute returned 0");

    /* C(3,1) = 3 subsets, each removes 1 vertex.
     * Each subgraph is a single edge (2 vertices, 1 edge) with dress = 2.0.
     * Total: 3 edges contributing to histogram. */
    int64_t total = 0;
    for (int i = 0; i < result.hist_size; i++)
        total += result.histogram[i];
    ASSERT_EQ_INT((int)total, 3, "Δ^1 triangle: 3 edge values");

    /* All 3 edge values should be 2.0 → top or second-to-top bin */
    int64_t top = result.histogram[result.hist_size - 1]
               + result.histogram[result.hist_size - 2];
    ASSERT_EQ_INT((int)top, 3, "all in top bins");

    delta_dress_free_igraph(&result);
    igraph_destroy(&g);
}

/* ── test: Δ^0 on K4 ──────────────────────────────────────────────── */

static void test_delta0_k4(void)
{
    printf("test_delta0_k4\n");

    int edges[] = {0,1, 0,2, 0,3, 1,2, 1,3, 2,3};
    igraph_t g = make_graph(edges, 6, 4, IGRAPH_UNDIRECTED);

    delta_dress_result_igraph_t result;
    int rc = delta_dress_fit_igraph(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                                        0, 100, 1e-6, 0, &result);

    ASSERT_EQ_INT(rc, 0, "compute returned 0");

    int64_t total = 0;
    for (int i = 0; i < result.hist_size; i++)
        total += result.histogram[i];
    ASSERT_EQ_INT((int)total, 6, "K4 Δ^0: 6 edge values");

    /* All edges in K4 have dress = 2.0 → top bins */
    int64_t top = result.histogram[result.hist_size - 1]
               + result.histogram[result.hist_size - 2];
    ASSERT_EQ_INT((int)top, 6, "all in top bins");

    delta_dress_free_igraph(&result);
    igraph_destroy(&g);
}

/* ── test: Δ^1 on K4 ──────────────────────────────────────────────── */

static void test_delta1_k4(void)
{
    printf("test_delta1_k4\n");

    int edges[] = {0,1, 0,2, 0,3, 1,2, 1,3, 2,3};
    igraph_t g = make_graph(edges, 6, 4, IGRAPH_UNDIRECTED);

    delta_dress_result_igraph_t result;
    int rc = delta_dress_fit_igraph(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                                        1, 100, 1e-6, 0, &result);

    ASSERT_EQ_INT(rc, 0, "compute returned 0");

    /* C(4,1) = 4 subsets.  Removing any vertex from K4 gives K3 (3 edges,
     * each with dress 2.0).  Total: 4 × 3 = 12 edge values, all = 2.0. */
    int64_t total = 0;
    for (int i = 0; i < result.hist_size; i++)
        total += result.histogram[i];
    ASSERT_EQ_INT((int)total, 12, "Δ^1 K4: 12 edge values");
    int64_t top = result.histogram[result.hist_size - 1]
               + result.histogram[result.hist_size - 2];
    ASSERT_EQ_INT((int)top, 12, "all in top bins");

    delta_dress_free_igraph(&result);
    igraph_destroy(&g);
}

/* ── test: empty graph produces zero histogram ─────────────────────── */

static void test_delta_empty_graph(void)
{
    printf("test_delta_empty_graph\n");

    igraph_t g;
    igraph_empty(&g, 5, IGRAPH_UNDIRECTED);

    delta_dress_result_igraph_t result;
    int rc = delta_dress_fit_igraph(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                                        0, 100, 1e-6, 0, &result);

    ASSERT_EQ_INT(rc, 0, "empty graph returns 0");
    ASSERT_GT(result.hist_size, 0, "hist_size > 0");

    int64_t total = 0;
    for (int i = 0; i < result.hist_size; i++)
        total += result.histogram[i];
    ASSERT_EQ_INT((int)total, 0, "empty graph: 0 edge values");

    delta_dress_free_igraph(&result);
    igraph_destroy(&g);
}

/* ── test: null params ─────────────────────────────────────────────── */

static void test_delta_null_params(void)
{
    printf("test_delta_null_params\n");

    delta_dress_result_igraph_t result;
    int rc;

    rc = delta_dress_fit_igraph(NULL, NULL, DRESS_VARIANT_UNDIRECTED,
                                    0, 100, 1e-6, 0, &result);
    ASSERT(rc != 0, "NULL graph returns error");

    igraph_t g;
    igraph_empty(&g, 3, IGRAPH_UNDIRECTED);

    rc = delta_dress_fit_igraph(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                                    0, 100, 1e-6, 0, NULL);
    ASSERT(rc != 0, "NULL result returns error");

    igraph_destroy(&g);
}

/* ── test: delta_to_vector ─────────────────────────────────────────── */

static void test_delta_to_vector(void)
{
    printf("test_delta_to_vector\n");

    int edges[] = {0,1, 1,2, 0,2};
    igraph_t g = make_graph(edges, 3, 3, IGRAPH_UNDIRECTED);

    delta_dress_result_igraph_t result;
    delta_dress_fit_igraph(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                               0, 100, 1e-6, 0, &result);

    igraph_vector_t vec;
    igraph_vector_init(&vec, 0);
    int rc = delta_dress_to_vector_igraph(&result, &vec);

    ASSERT_EQ_INT(rc, 0, "to_vector returned 0");
    ASSERT_EQ_INT((int)igraph_vector_size(&vec), result.hist_size,
                  "vector size == hist_size");

    int vec_match = 1;
    for (int i = 0; i < result.hist_size; i++) {
        if (VECTOR(vec)[i] != (double)result.histogram[i]) {
            vec_match = 0; break;
        }
    }
    ASSERT(vec_match, "vector matches histogram for all bins");

    igraph_vector_destroy(&vec);
    delta_dress_free_igraph(&result);
    igraph_destroy(&g);
}

/* ── test: cross-validate with raw C delta_fit ─────────────────────── */

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

    delta_dress_result_igraph_t ig_result;
    delta_dress_fit_igraph(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                               1, 100, 1e-6, 0, &ig_result);

    /* Build via raw C API with the same edge list */
    int src[] = {0, 1, 2, 0};
    int dst[] = {1, 2, 3, 2};
    p_dress_graph_t dg = init_dress_graph(
        4, 4, dup_int_arr(src, 4), dup_int_arr(dst, 4),
        NULL, DRESS_VARIANT_UNDIRECTED, 0);

    int hist_size = 0;
    int64_t *histogram = delta_dress_fit(dg, 1, 100, 1e-6, &hist_size, 0, NULL, NULL);
    free_dress_graph(dg);

    /* Compare histogram sizes */
    ASSERT_EQ_INT(ig_result.hist_size, hist_size, "same hist_size");

    /* Compare every bin */
    ASSERT(memcmp(ig_result.histogram, histogram,
                  (size_t)hist_size * sizeof(int64_t)) == 0,
           "igraph matches raw C delta histogram for all bins");

    free(histogram);
    delta_dress_free_igraph(&ig_result);
    igraph_destroy(&g);
}

/* ── test: precompute flag ─────────────────────────────────────────── */

static void test_delta_precompute(void)
{
    printf("test_delta_precompute\n");

    int edges[] = {0,1, 1,2, 0,2};
    igraph_t g = make_graph(edges, 3, 3, IGRAPH_UNDIRECTED);

    delta_dress_result_igraph_t r_no, r_yes;

    delta_dress_fit_igraph(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                               1, 100, 1e-6, 0, &r_no);
    delta_dress_fit_igraph(&g, NULL, DRESS_VARIANT_UNDIRECTED,
                               1, 100, 1e-6, 1, &r_yes);

    ASSERT_EQ_INT(r_no.hist_size, r_yes.hist_size, "same hist_size");
    ASSERT(memcmp(r_no.histogram, r_yes.histogram,
                  (size_t)r_no.hist_size * sizeof(int64_t)) == 0,
           "precompute gives same histogram for all bins");

    delta_dress_free_igraph(&r_no);
    delta_dress_free_igraph(&r_yes);
    igraph_destroy(&g);
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(void)
{
    igraph_set_attribute_table(&igraph_cattribute_table);

    printf("=== Δ^k-DRESS igraph tests ===\n\n");

    test_delta0_triangle();
    test_delta1_triangle();
    test_delta0_k4();
    test_delta1_k4();
    test_delta_empty_graph();
    test_delta_null_params();
    test_delta_to_vector();
    test_cross_validate_with_c_api();
    test_delta_precompute();

    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
