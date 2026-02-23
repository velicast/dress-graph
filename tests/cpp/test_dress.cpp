/*
 * Tests for the DRESS C++ wrapper (libdress++).
 *
 * Build from the repo root:
 *   g++ -std=c++17 -O2 -I libdress/include -I libdress++/include \
 *       -o tests/cpp/test_dress \
 *       tests/cpp/test_dress.cpp libdress/src/dress.c -lm -fopenmp
 *
 * Run:
 *   ./tests/cpp/test_dress
 */

#include "dress/dress.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

/* ── tiny test harness ─────────────────────────────────────────────── */

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT(cond, msg)                                                \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n",                   \
                         __FILE__, __LINE__, (msg));                      \
            g_fail++;                                                    \
        } else {                                                         \
            g_pass++;                                                    \
        }                                                                \
    } while (0)

#define ASSERT_EQ(a, b, msg)        ASSERT((a) == (b), msg)
#define ASSERT_GT(a, b, msg)        ASSERT((a) >  (b), msg)
#define ASSERT_LT(a, b, msg)        ASSERT((a) <  (b), msg)
#define ASSERT_NEAR(a, b, tol, msg) ASSERT(std::fabs((a) - (b)) < (tol), msg)

#define ASSERT_THROWS(expr, exc_type, msg)                               \
    do {                                                                 \
        bool caught = false;                                             \
        try { expr; } catch (const exc_type&) { caught = true; }        \
        ASSERT(caught, msg);                                             \
    } while (0)

/* ── test: construction ────────────────────────────────────────────── */

static void test_unweighted_triangle()
{
    std::printf("test_unweighted_triangle\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});

    ASSERT_EQ(g.numVertices(), 3, "N == 3");
    ASSERT_EQ(g.numEdges(),    3, "E == 3");
    ASSERT_EQ(g.variant(), DRESS_VARIANT_UNDIRECTED, "variant == UNDIRECTED");

    ASSERT_EQ(g.edgeSource(0), 0, "U[0] == 0");
    ASSERT_EQ(g.edgeTarget(0), 1, "V[0] == 1");
    ASSERT_EQ(g.edgeSource(1), 1, "U[1] == 1");
    ASSERT_EQ(g.edgeTarget(1), 2, "V[1] == 2");
}

static void test_weighted_construction()
{
    std::printf("test_weighted_construction\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2}, {1.0, 2.0, 3.0});

    ASSERT_EQ(g.numEdges(), 3, "E == 3");
    /* Undirected weighted edges get doubled */
    ASSERT_NEAR(g.edgeWeight(0), 2.0, 1e-12, "w[0] == 2.0");
    ASSERT_NEAR(g.edgeWeight(1), 4.0, 1e-12, "w[1] == 4.0");
    ASSERT_NEAR(g.edgeWeight(2), 6.0, 1e-12, "w[2] == 6.0");
}

static void test_all_variants()
{
    std::printf("test_all_variants\n");

    dress_variant_t variants[] = {
        DRESS_VARIANT_UNDIRECTED,
        DRESS_VARIANT_DIRECTED,
        DRESS_VARIANT_FORWARD,
        DRESS_VARIANT_BACKWARD
    };

    for (auto v : variants) {
        DRESS g(3, {0, 1, 0}, {1, 2, 2}, v);
        ASSERT_EQ(g.variant(), v, "variant stored correctly");
        ASSERT_EQ(g.numEdges(), 3, "E == 3");
    }
}

static void test_precompute_intercepts()
{
    std::printf("test_precompute_intercepts\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2},
            DRESS_VARIANT_UNDIRECTED, /*precompute_intercepts=*/true);

    ASSERT_EQ(g.numEdges(), 3, "E == 3");
    ASSERT(g.raw()->precompute_intercepts == 1, "intercepts enabled");
    ASSERT(g.raw()->intercept_offset != nullptr, "intercept_offset allocated");
}

static void test_mismatched_sizes()
{
    std::printf("test_mismatched_sizes\n");

    ASSERT_THROWS(
        DRESS(3, {0, 1}, {1, 2, 2}),
        std::invalid_argument,
        "U/V size mismatch throws invalid_argument");

    ASSERT_THROWS(
        DRESS(3, {0, 1, 0}, {1, 2, 2}, {1.0, 2.0}),
        std::invalid_argument,
        "U/W size mismatch throws invalid_argument");
}

/* ── test: move semantics ──────────────────────────────────────────── */

static void test_move_constructor()
{
    std::printf("test_move_constructor\n");

    DRESS g1(3, {0, 1, 0}, {1, 2, 2});
    DRESS g2(std::move(g1));

    ASSERT_EQ(g2.numVertices(), 3, "moved-to has N == 3");
    ASSERT_EQ(g2.numEdges(),    3, "moved-to has E == 3");

    /* Accessing moved-from should throw. */
    ASSERT_THROWS(
        g1.numVertices(),
        std::logic_error,
        "accessing moved-from graph throws logic_error");
}

static void test_move_assignment()
{
    std::printf("test_move_assignment\n");

    DRESS g1(3, {0, 1, 0}, {1, 2, 2});
    DRESS g2(2, {0}, {1});

    g2 = std::move(g1);

    ASSERT_EQ(g2.numVertices(), 3, "assigned-to has N == 3");
    ASSERT_THROWS(
        g1.numEdges(),
        std::logic_error,
        "moved-from assignment throws");
}

/* ── test: fitting ─────────────────────────────────────────────────── */

static void test_triangle_convergence()
{
    std::printf("test_triangle_convergence\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});
    auto [iters, delta] = g.fit(100, 1e-8);

    ASSERT_GT(iters, 0, "iterations > 0");
    ASSERT(delta >= 0.0, "delta >= 0");
}

static void test_triangle_equal_dress()
{
    std::printf("test_triangle_equal_dress\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});
    g.fit(100, 1e-8);

    double d0 = g.edgeDress(0);
    ASSERT_NEAR(g.edgeDress(1), d0, 1e-6, "edge 1 == edge 0");
    ASSERT_NEAR(g.edgeDress(2), d0, 1e-6, "edge 2 == edge 0");
}

static void test_path_positive_dress()
{
    std::printf("test_path_positive_dress\n");

    DRESS g(4, {0, 1, 2}, {1, 2, 3});
    g.fit(100, 1e-6);

    for (int e = 0; e < g.numEdges(); e++) {
        ASSERT_GT(g.edgeDress(e), 0.0, "path dress > 0 (self-loop term)");
        ASSERT_LT(g.edgeDress(e), 2.0, "path dress < 2");
    }
}

static void test_path_symmetry()
{
    std::printf("test_path_symmetry\n");

    DRESS g(4, {0, 1, 2}, {1, 2, 3});
    g.fit(100, 1e-6);

    ASSERT_NEAR(g.edgeDress(0), g.edgeDress(2), 1e-10,
                "endpoint edges symmetric");
}

static void test_fit_with_intercepts()
{
    std::printf("test_fit_with_intercepts\n");

    DRESS g1(3, {0, 1, 0}, {1, 2, 2},
             DRESS_VARIANT_UNDIRECTED, /*precompute=*/true);
    DRESS g2(3, {0, 1, 0}, {1, 2, 2},
             DRESS_VARIANT_UNDIRECTED, /*precompute=*/false);

    g1.fit(100, 1e-10);
    g2.fit(100, 1e-10);

    for (int e = 0; e < 3; e++) {
        ASSERT_NEAR(g1.edgeDress(e), g2.edgeDress(e), 1e-8,
                    "intercept path matches no-intercept");
    }
}

static void test_node_dress()
{
    std::printf("test_node_dress\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});
    g.fit(100, 1e-8);

    ASSERT_GT(g.nodeDress(0), 0.0, "node_dress[0] > 0");
    ASSERT_NEAR(g.nodeDress(0), g.nodeDress(1), 1e-6,
                "K3: all node dress equal (0 vs 1)");
    ASSERT_NEAR(g.nodeDress(0), g.nodeDress(2), 1e-6,
                "K3: all node dress equal (0 vs 2)");
}

static void test_weighted_fit()
{
    std::printf("test_weighted_fit\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2}, {1.0, 2.0, 3.0});
    auto [iters, delta] = g.fit(100, 1e-6);

    ASSERT_GT(iters, 0, "weighted iterations > 0");

    double d_min = g.edgeDress(0), d_max = g.edgeDress(0);
    for (int e = 1; e < 3; e++) {
        if (g.edgeDress(e) < d_min) d_min = g.edgeDress(e);
        if (g.edgeDress(e) > d_max) d_max = g.edgeDress(e);
    }
    ASSERT_GT(d_max - d_min, 1e-6,
              "asymmetric weights produce different dress values");
}

/* ── test: bulk accessors ──────────────────────────────────────────── */

static void test_bulk_accessors()
{
    std::printf("test_bulk_accessors\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});
    g.fit(100, 1e-8);

    const int    *src = g.edgeSources();
    const int    *tgt = g.edgeTargets();
    const double *wts = g.edgeWeights();
    const double *drv = g.edgeDressValues();
    const double *ndv = g.nodeDressValues();

    ASSERT(src != nullptr, "edgeSources not null");
    ASSERT(tgt != nullptr, "edgeTargets not null");
    ASSERT(wts != nullptr, "edgeWeights not null");
    ASSERT(drv != nullptr, "edgeDressValues not null");
    ASSERT(ndv != nullptr, "nodeDressValues not null");

    ASSERT_EQ(src[0], 0, "bulk src[0] == 0");
    ASSERT_EQ(tgt[0], 1, "bulk tgt[0] == 1");
    ASSERT_GT(drv[0], 0.0, "bulk dress > 0 after fit");
}

static void test_csr_accessors()
{
    std::printf("test_csr_accessors\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});

    const int *off = g.adjOffset();
    const int *adj = g.adjTarget();
    const int *eidx = g.adjEdgeIdx();

    ASSERT(off != nullptr, "adjOffset not null");
    ASSERT(adj != nullptr, "adjTarget not null");
    ASSERT(eidx != nullptr, "adjEdgeIdx not null");

    /* CSR offsets: off[0] == 0, off[N] == total half-edges */
    ASSERT_EQ(off[0], 0, "CSR off[0] == 0");
    ASSERT_GT(off[3], 0, "CSR off[N] > 0 (has half-edges)");
}

/* ── test: edge cases ──────────────────────────────────────────────── */

static void test_single_edge()
{
    std::printf("test_single_edge\n");

    DRESS g(2, {0}, {1});
    g.fit(100, 1e-8);

    ASSERT_EQ(g.numVertices(), 2, "N == 2");
    ASSERT_EQ(g.numEdges(), 1, "E == 1");
    ASSERT_GT(g.edgeDress(0), 0.0, "single edge dress > 0");
}

static void test_complete_k4()
{
    std::printf("test_complete_k4\n");

    DRESS g(4, {0,0,0,1,1,2}, {1,2,3,2,3,3});
    g.fit(200, 1e-10);

    double d0 = g.edgeDress(0);
    for (int e = 1; e < 6; e++) {
        ASSERT_NEAR(g.edgeDress(e), d0, 1e-6, "K4 edges equal");
    }

    for (int u = 1; u < 4; u++) {
        ASSERT_NEAR(g.nodeDress(u), g.nodeDress(0), 1e-6,
                    "K4 nodes equal");
    }
}

static void test_star_graph()
{
    std::printf("test_star_graph\n");

    DRESS g(5, {0,0,0,0}, {1,2,3,4});
    g.fit(100, 1e-8);

    double d0 = g.edgeDress(0);
    for (int e = 1; e < 4; e++) {
        ASSERT_NEAR(g.edgeDress(e), d0, 1e-6, "star edges equal");
    }
}

static void test_raw_access()
{
    std::printf("test_raw_access\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});

    p_dress_graph_t raw = g.raw();
    ASSERT(raw != nullptr, "raw() not null");
    ASSERT_EQ(raw->N, 3, "raw N == 3");
    ASSERT_EQ(raw->E, 3, "raw E == 3");

    const dress_graph_t *craw = static_cast<const DRESS&>(g).raw();
    ASSERT(craw != nullptr, "const raw() not null");
}

/* ── main ──────────────────────────────────────────────────────────── */

int main()
{
    std::printf("=== DRESS C++ tests ===\n\n");

    /* construction */
    test_unweighted_triangle();
    test_weighted_construction();
    test_all_variants();
    test_precompute_intercepts();
    test_mismatched_sizes();

    /* move semantics */
    test_move_constructor();
    test_move_assignment();

    /* fitting */
    test_triangle_convergence();
    test_triangle_equal_dress();
    test_path_positive_dress();
    test_path_symmetry();
    test_fit_with_intercepts();
    test_node_dress();
    test_weighted_fit();

    /* accessors */
    test_bulk_accessors();
    test_csr_accessors();

    /* edge cases */
    test_single_edge();
    test_complete_k4();
    test_star_graph();
    test_raw_access();

    std::printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
