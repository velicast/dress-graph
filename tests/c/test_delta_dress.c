/*
 * Tests for the Δ^k-DRESS C library (delta_dress).
 *
 * Build from the repo root:
 *   gcc -O2 -I libdress/include -o tests/c/test_delta_dress \
 *       tests/c/test_delta_dress.c libdress/src/dress.c \
 *       libdress/src/delta_dress.c -lm -fopenmp
 *
 * Run:
 *   ./tests/c/test_delta_dress
 */

#include "dress/dress.h"
#include "dress/delta_dress.h"

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

#define ASSERT_EQ(a, b, msg)        ASSERT((a) == (b), msg)
#define ASSERT_GT(a, b, msg)        ASSERT((a) >  (b), msg)

/* ── helpers ───────────────────────────────────────────────────────── */

/* Allocate a malloc'd copy of a stack array (C API takes ownership). */
static int *dup_int(const int *src, int n)
{
    int *p = (int *)malloc((size_t)n * sizeof(int));
    memcpy(p, src, (size_t)n * sizeof(int));
    return p;
}

/* Sum all entries in a histogram. */
static int64_t hist_total(const int64_t *hist, int nbins)
{
    int64_t total = 0;
    for (int i = 0; i < nbins; i++) total += hist[i];
    return total;
}

/* ── test: histogram size ──────────────────────────────────────────── */

static void test_hist_size(void)
{
    printf("test_hist_size\n");

    int U[] = {0, 1, 0};
    int V[] = {1, 2, 2};
    p_dress_graph_t g = init_dress_graph(3, 3, dup_int(U, 3), dup_int(V, 3),
                                         NULL, DRESS_VARIANT_UNDIRECTED, 0);
    int hsize;

    /* eps = 1e-3 → floor(2/1e-3) + 1 = 2001 */
    int64_t *h = delta_fit(g, 0, 100, 1e-3, &hsize);
    ASSERT_EQ(hsize, 2001, "hist_size == 2001 for eps=1e-3");
    free(h);

    /* eps = 1e-6 → floor(2/1e-6) + 1 = 2000001 */
    h = delta_fit(g, 0, 100, 1e-6, &hsize);
    ASSERT_EQ(hsize, 2000001, "hist_size == 2000001 for eps=1e-6");
    free(h);

    free_dress_graph(g);
}

/* ── test: Δ^0 on K3 ──────────────────────────────────────────────── */

static void test_delta0_k3(void)
{
    printf("test_delta0_k3\n");

    int U[] = {0, 1, 0};
    int V[] = {1, 2, 2};
    p_dress_graph_t g = init_dress_graph(3, 3, dup_int(U, 3), dup_int(V, 3),
                                         NULL, DRESS_VARIANT_UNDIRECTED, 0);
    int hsize;
    int64_t *h = delta_fit(g, 0, 100, 1e-3, &hsize);

    /* K3 is vertex-transitive → all edges equal → single non-zero bin */
    int nonzero = 0;
    for (int i = 0; i < hsize; i++)
        if (h[i] > 0) nonzero++;
    ASSERT_EQ(nonzero, 1, "delta0 K3: all edges in same bin");

    /* The bin should be 2000 (value 2.0 — complete symmetry) */
    ASSERT_GT(h[hsize - 1], 0, "delta0 K3: top bin holds value 2.0");

    free(h);
    free_dress_graph(g);
}

/* ── test: Δ^1 on K3 ──────────────────────────────────────────────── */

static void test_delta1_k3(void)
{
    printf("test_delta1_k3\n");

    int U[] = {0, 1, 0};
    int V[] = {1, 2, 2};
    p_dress_graph_t g = init_dress_graph(3, 3, dup_int(U, 3), dup_int(V, 3),
                                         NULL, DRESS_VARIANT_UNDIRECTED, 0);
    int hsize;
    int64_t *h = delta_fit(g, 1, 100, 1e-3, &hsize);

    /* C(3,1) = 3 subgraphs, each has 1 edge */
    ASSERT_EQ(hist_total(h, hsize), 3,
              "delta1 K3: 3 subgraphs * 1 edge = 3");

    free(h);
    free_dress_graph(g);
}

/* ── test: Δ^2 on K3 → zero edges ─────────────────────────────────── */

static void test_delta2_k3(void)
{
    printf("test_delta2_k3\n");

    int U[] = {0, 1, 0};
    int V[] = {1, 2, 2};
    p_dress_graph_t g = init_dress_graph(3, 3, dup_int(U, 3), dup_int(V, 3),
                                         NULL, DRESS_VARIANT_UNDIRECTED, 0);
    int hsize;
    int64_t *h = delta_fit(g, 2, 100, 1e-3, &hsize);

    /* Removing 2 of 3 vertices leaves 1 vertex, 0 edges */
    ASSERT_EQ(hist_total(h, hsize), 0, "delta2 K3: 0 edge values");

    free(h);
    free_dress_graph(g);
}

/* ── test: Δ^0 on K4 ──────────────────────────────────────────────── */

static void test_delta0_k4(void)
{
    printf("test_delta0_k4\n");

    int U[] = {0, 0, 0, 1, 1, 2};
    int V[] = {1, 2, 3, 2, 3, 3};
    p_dress_graph_t g = init_dress_graph(4, 6, dup_int(U, 6), dup_int(V, 6),
                                         NULL, DRESS_VARIANT_UNDIRECTED, 0);
    int hsize;
    int64_t *h = delta_fit(g, 0, 100, 1e-3, &hsize);

    ASSERT_EQ(hist_total(h, hsize), 6, "delta0 K4: 6 edge values");

    /* K4 is vertex-transitive → all 6 edges at value 2.0 */
    ASSERT_EQ(h[hsize - 1], 6, "delta0 K4: all edges at top bin");

    free(h);
    free_dress_graph(g);
}

/* ── test: Δ^1 on K4 ──────────────────────────────────────────────── */

static void test_delta1_k4(void)
{
    printf("test_delta1_k4\n");

    int U[] = {0, 0, 0, 1, 1, 2};
    int V[] = {1, 2, 3, 2, 3, 3};
    p_dress_graph_t g = init_dress_graph(4, 6, dup_int(U, 6), dup_int(V, 6),
                                         NULL, DRESS_VARIANT_UNDIRECTED, 0);
    int hsize;
    int64_t *h = delta_fit(g, 1, 100, 1e-3, &hsize);

    /* C(4,1) = 4 subgraphs, each K3 with 3 edges */
    ASSERT_EQ(hist_total(h, hsize), 12,
              "delta1 K4: 4 * 3 = 12 edge values");

    /* Each subgraph is K3 → all edges at 2.0 */
    ASSERT_EQ(h[hsize - 1], 12,
              "delta1 K4: all 12 edges at top bin");

    free(h);
    free_dress_graph(g);
}

/* ── test: Δ^2 on K4 ──────────────────────────────────────────────── */

static void test_delta2_k4(void)
{
    printf("test_delta2_k4\n");

    int U[] = {0, 0, 0, 1, 1, 2};
    int V[] = {1, 2, 3, 2, 3, 3};
    p_dress_graph_t g = init_dress_graph(4, 6, dup_int(U, 6), dup_int(V, 6),
                                         NULL, DRESS_VARIANT_UNDIRECTED, 0);
    int hsize;
    int64_t *h = delta_fit(g, 2, 100, 1e-3, &hsize);

    /* C(4,2) = 6 subgraphs, each has 1 edge */
    ASSERT_EQ(hist_total(h, hsize), 6,
              "delta2 K4: 6 * 1 = 6 edge values");

    free(h);
    free_dress_graph(g);
}

/* ── test: k >= N returns empty histogram ──────────────────────────── */

static void test_k_ge_N(void)
{
    printf("test_k_ge_N\n");

    int U[] = {0, 1, 0};
    int V[] = {1, 2, 2};
    p_dress_graph_t g = init_dress_graph(3, 3, dup_int(U, 3), dup_int(V, 3),
                                         NULL, DRESS_VARIANT_UNDIRECTED, 0);
    int hsize;

    int64_t *h = delta_fit(g, 3, 100, 1e-3, &hsize);
    ASSERT_EQ(hist_total(h, hsize), 0, "k == N: empty histogram");
    free(h);

    h = delta_fit(g, 10, 100, 1e-3, &hsize);
    ASSERT_EQ(hist_total(h, hsize), 0, "k > N: empty histogram");
    free(h);

    free_dress_graph(g);
}

/* ── test: precompute flag ─────────────────────────────────────────── */

static void test_precompute(void)
{
    printf("test_precompute\n");

    int U[] = {0, 0, 0, 1, 1, 2};
    int V[] = {1, 2, 3, 2, 3, 3};

    /* precompute = 0 */
    p_dress_graph_t g1 = init_dress_graph(4, 6, dup_int(U, 6), dup_int(V, 6),
                                          NULL, DRESS_VARIANT_UNDIRECTED, 0);
    /* precompute = 1 */
    p_dress_graph_t g2 = init_dress_graph(4, 6, dup_int(U, 6), dup_int(V, 6),
                                          NULL, DRESS_VARIANT_UNDIRECTED, 1);
    int hsize1, hsize2;

    int64_t *h1 = delta_fit(g1, 1, 100, 1e-3, &hsize1);
    int64_t *h2 = delta_fit(g2, 1, 100, 1e-3, &hsize2);

    ASSERT_EQ(hsize1, hsize2, "precompute: same hist size");
    int match = 1;
    for (int i = 0; i < hsize1; i++) {
        if (h1[i] != h2[i]) { match = 0; break; }
    }
    ASSERT(match, "precompute: identical histograms");

    free(h1);
    free(h2);
    free_dress_graph(g1);
    free_dress_graph(g2);
}

/* ── test: path graph P4 — edge values differ ─────────────────────── */

static void test_delta0_path(void)
{
    printf("test_delta0_path\n");

    /* P4: 0-1-2-3 */
    int U[] = {0, 1, 2};
    int V[] = {1, 2, 3};
    p_dress_graph_t g = init_dress_graph(4, 3, dup_int(U, 3), dup_int(V, 3),
                                         NULL, DRESS_VARIANT_UNDIRECTED, 0);
    int hsize;
    int64_t *h = delta_fit(g, 0, 100, 1e-3, &hsize);

    ASSERT_EQ(hist_total(h, hsize), 3, "delta0 P4: 3 edge values");

    /* P4 is NOT vertex-transitive: edges (0,1) and (2,3) are peripheral,
       edge (1,2) is central → at least 2 distinct bins */
    int nonzero = 0;
    for (int i = 0; i < hsize; i++)
        if (h[i] > 0) nonzero++;
    ASSERT_GT(nonzero, 1, "delta0 P4: edges not all equal");

    free(h);
    free_dress_graph(g);
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(void)
{
    test_hist_size();
    test_delta0_k3();
    test_delta1_k3();
    test_delta2_k3();
    test_delta0_k4();
    test_delta1_k4();
    test_delta2_k4();
    test_k_ge_N();
    test_precompute();
    test_delta0_path();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
