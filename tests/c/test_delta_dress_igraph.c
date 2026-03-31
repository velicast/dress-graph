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
   C API, so we need the un-redirected dress_delta_fit() symbol. */
#undef dress_delta_fit


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

static int64_t hist_total(const delta_dress_result_igraph_t *result)
{
    int64_t total = 0;

    for (int i = 0; i < result->hist_size; i++)
        total += result->histogram[i].count;

    return total;
}

static int64_t hist_count_value(const delta_dress_result_igraph_t *result,
                                double value)
{
    int64_t total = 0;

    for (int i = 0; i < result->hist_size; i++) {
        if (fabs(result->histogram[i].value - value) < 1e-9)
            total += result->histogram[i].count;
    }

    return total;
}

static int hist_equal(const dress_hist_pair_t *a, int a_size,
                      const dress_hist_pair_t *b, int b_size)
{
    if (a_size != b_size)
        return 0;

    for (int i = 0; i < a_size; i++) {
        if (fabs(a[i].value - b[i].value) >= 1e-9 || a[i].count != b[i].count)
            return 0;
    }

    return 1;
}

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
    int rc = dress_delta_fit_igraph(&g, NULL, NULL, DRESS_VARIANT_UNDIRECTED,
                                        0, 100, 1e-6, 0, 0, 0, 0, 1, &result);

    ASSERT_EQ_INT(rc, 0, "delta compute returned 0");
    ASSERT_GT(result.hist_size, 0, "hist_size > 0");
    ASSERT(result.histogram != NULL, "histogram not NULL");

    ASSERT_EQ_INT((int)hist_total(&result), 3, "Δ^0 triangle: sum == 3 edges");
    ASSERT_EQ_INT(result.hist_size, 1, "Δ^0 triangle: one exact histogram entry");
    ASSERT_EQ_INT((int)hist_count_value(&result, 2.0), 3,
                  "Δ^0 triangle: value 2.0 count == 3");

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
    int rc = dress_delta_fit_igraph(&g, NULL, NULL, DRESS_VARIANT_UNDIRECTED,
                                        1, 100, 1e-6, 0, 0, 0, 0, 1, &result);

    ASSERT_EQ_INT(rc, 0, "delta compute returned 0");

    /* C(3,1) = 3 subsets, each removes 1 vertex.
     * Each subgraph is a single edge (2 vertices, 1 edge) with dress = 2.0.
     * Total: 3 edges contributing to histogram. */
    ASSERT_EQ_INT((int)hist_total(&result), 3, "Δ^1 triangle: 3 edge values");
    ASSERT_EQ_INT(result.hist_size, 1, "Δ^1 triangle: one exact histogram entry");
    ASSERT_EQ_INT((int)hist_count_value(&result, 2.0), 3,
                  "Δ^1 triangle: value 2.0 count == 3");

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
    int rc = dress_delta_fit_igraph(&g, NULL, NULL, DRESS_VARIANT_UNDIRECTED,
                                        0, 100, 1e-6, 0, 0, 0, 0, 1, &result);

    ASSERT_EQ_INT(rc, 0, "compute returned 0");

    ASSERT_EQ_INT((int)hist_total(&result), 6, "K4 Δ^0: 6 edge values");
    ASSERT_EQ_INT(result.hist_size, 1, "K4 Δ^0: one exact histogram entry");
    ASSERT_EQ_INT((int)hist_count_value(&result, 2.0), 6,
                  "K4 Δ^0: value 2.0 count == 6");

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
    int rc = dress_delta_fit_igraph(&g, NULL, NULL, DRESS_VARIANT_UNDIRECTED,
                                        1, 100, 1e-6, 0, 0, 0, 0, 1, &result);

    ASSERT_EQ_INT(rc, 0, "compute returned 0");

    /* C(4,1) = 4 subsets.  Removing any vertex from K4 gives K3 (3 edges,
     * each with dress 2.0).  Total: 4 × 3 = 12 edge values, all = 2.0. */
    ASSERT_EQ_INT((int)hist_total(&result), 12, "Δ^1 K4: 12 edge values");
    ASSERT_EQ_INT(result.hist_size, 1, "Δ^1 K4: one exact histogram entry");
    ASSERT_EQ_INT((int)hist_count_value(&result, 2.0), 12,
                  "Δ^1 K4: value 2.0 count == 12");

    delta_dress_free_igraph(&result);
    igraph_destroy(&g);
}

/* ── test: empty graph produces empty histogram ────────────────────── */

static void test_delta_empty_graph(void)
{
    printf("test_delta_empty_graph\n");

    igraph_t g;
    igraph_empty(&g, 5, IGRAPH_UNDIRECTED);

    delta_dress_result_igraph_t result;
    int rc = dress_delta_fit_igraph(&g, NULL, NULL, DRESS_VARIANT_UNDIRECTED,
                                        0, 100, 1e-6, 0, 0, 0, 0, 1, &result);

    ASSERT_EQ_INT(rc, 0, "empty graph returns 0");
    ASSERT_EQ_INT(result.hist_size, 0, "empty graph: 0 histogram entries");
    ASSERT(result.histogram == NULL, "empty graph: histogram is NULL");
    ASSERT_EQ_INT((int)hist_total(&result), 0, "empty graph: 0 edge values");

    delta_dress_free_igraph(&result);
    igraph_destroy(&g);
}

/* ── test: null params ─────────────────────────────────────────────── */

static void test_delta_null_params(void)
{
    printf("test_delta_null_params\n");

    delta_dress_result_igraph_t result;
    int rc;

    rc = dress_delta_fit_igraph(NULL, NULL, NULL, DRESS_VARIANT_UNDIRECTED,
                                    0, 100, 1e-6, 0, 0, 0, 0, 1, &result);
    ASSERT(rc != 0, "NULL graph returns error");

    igraph_t g;
    igraph_empty(&g, 3, IGRAPH_UNDIRECTED);

    rc = dress_delta_fit_igraph(&g, NULL, NULL, DRESS_VARIANT_UNDIRECTED,
                                    0, 100, 1e-6, 0, 0, 0, 0, 1, NULL);
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
    dress_delta_fit_igraph(&g, NULL, NULL, DRESS_VARIANT_UNDIRECTED,
                               0, 100, 1e-6, 0, 0, 0, 0, 1, &result);

    igraph_vector_t vec;
    igraph_vector_init(&vec, 0);
    int rc = delta_dress_to_vector_igraph(&result, &vec);

    ASSERT_EQ_INT(rc, 0, "to_vector returned 0");
    ASSERT_EQ_INT((int)igraph_vector_size(&vec), 2 * result.hist_size,
                  "vector size == 2 * hist_size");

    int vec_match = 1;
    for (int i = 0; i < result.hist_size; i++) {
        if (VECTOR(vec)[2 * i] != result.histogram[i].value ||
            VECTOR(vec)[2 * i + 1] != (double)result.histogram[i].count) {
            vec_match = 0; break;
        }
    }
    ASSERT(vec_match, "vector matches histogram value/count pairs");

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
    dress_delta_fit_igraph(&g, NULL, NULL, DRESS_VARIANT_UNDIRECTED,
                               1, 100, 1e-6, 0, 0, 0, 0, 1, &ig_result);

    /* Build via raw C API with the same edge list */
    int src[] = {0, 1, 2, 0};
    int dst[] = {1, 2, 3, 2};
    p_dress_graph_t dg = dress_init_graph(
        4, 4, dup_int_arr(src, 4), dup_int_arr(dst, 4),
        NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);

        int hist_size = 0;
        dress_hist_pair_t *histogram = dress_delta_fit(dg, 1, 100, 1e-6, 0, 0, &hist_size, 0, NULL, NULL);
    dress_free_graph(dg);

    ASSERT_EQ_INT(ig_result.hist_size, hist_size, "same hist_size");
        ASSERT(hist_equal(ig_result.histogram, ig_result.hist_size,
                 histogram, hist_size),
            "igraph matches raw C delta histogram for all entries");

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

    dress_delta_fit_igraph(&g, NULL, NULL, DRESS_VARIANT_UNDIRECTED,
                               1, 100, 1e-6, 0, 0, 0, 0, 1, &r_no);
    dress_delta_fit_igraph(&g, NULL, NULL, DRESS_VARIANT_UNDIRECTED,
                               1, 100, 1e-6, 0, 0, 1, 0, 1, &r_yes);

        ASSERT_EQ_INT(r_no.hist_size, r_yes.hist_size, "same hist_size");
        ASSERT(hist_equal(r_no.histogram, r_no.hist_size,
                 r_yes.histogram, r_yes.hist_size),
            "precompute gives same histogram for all entries");

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
