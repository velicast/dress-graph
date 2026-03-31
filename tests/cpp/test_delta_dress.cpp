/*
 * Tests for the Δ^k-DRESS C++ wrapper (libdress++).
 *
 * Build from the repo root:
 *   g++ -std=c++17 -O2 -I libdress/include -I libdress++/include \
 *       -o tests/cpp/test_delta_dress \
 *       tests/cpp/test_delta_dress.cpp libdress/src/dress.c \
 *       libdress/src/delta_dress.c -lm -fopenmp
 *
 * Run:
 *   ./tests/cpp/test_delta_dress
 */

#include "dress/dress.hpp"
using namespace dress;

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
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

#define ASSERT_EQ(a, b, msg) ASSERT((a) == (b), msg)
#define ASSERT_GT(a, b, msg) ASSERT((a) >  (b), msg)

static int64_t hist_total(const std::vector<std::pair<double, int64_t>> &h)
{
    int64_t sum = 0;
    for (auto const& [key, val] : h) {
        sum += val;
    }
    return sum;
}

static bool hist_equal(const std::vector<std::pair<double, int64_t>> &lhs,
                       const std::vector<std::pair<double, int64_t>> &rhs)
{
    return lhs == rhs;
}

/* ── test: histogram size ──────────────────────────────────────────── */

static void test_hist_size()
{
    std::printf("test_hist_size\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});

    // For K3 delta0 (k=0), all edges have same dress value (2.0).
    // So map size should be 1, regardless of epsilon (as long as it fits).
    
    auto r1 = g.deltaFit(0, 100, 1e-3);
    ASSERT_EQ((int)r1.histogram.size(), 1, "hist size == 1 (unique value)");
    ASSERT_EQ(hist_total(r1.histogram), 3, "total count == 3");

    auto r2 = g.deltaFit(0, 100, 1e-6);
    ASSERT_EQ((int)r2.histogram.size(), 1, "hist size == 1 (unique value)");
    ASSERT_EQ(hist_total(r2.histogram), 3, "total count == 3");
}

/* ── test: weighted histogram bin size ─────────────────────────────── */

static void test_weighted_hist_size()
{
    std::printf("test_weighted_hist_size\n");

    // Weighted K3: edges will have different values.
    DRESS g(3, {0, 1, 0}, {1, 2, 2}, {1.0, 10.0, 1.0});

    auto r = g.deltaFit(0, 100, 1e-3);
    
    // Expect > 1 unique values.
    ASSERT_GT((int)r.histogram.size(), 1,
              "weighted hist size > 1 (multiple values)");
    
    ASSERT_EQ(hist_total(r.histogram), 3,
              "weighted K3 delta0 total = 3");
}

/* ── test: Δ^0 on K3 ──────────────────────────────────────────────── */

static void test_delta0_k3()
{
    std::printf("test_delta0_k3\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});
    auto r = g.deltaFit(0, 100, 1e-3);

    ASSERT_EQ(hist_total(r.histogram), 3, "delta0 K3: 3 edge values");

    // K3: all edges equal → single non-zero bin (entry)
    ASSERT_EQ((int)r.histogram.size(), 1, "delta0 K3: all edges have same value");
}


/* ── test: no histogram (external rounding workflow) ───────────────── */

static void test_no_histogram()
{
    std::printf("test_no_histogram\n");
    // K3
    std::vector<int> U = {0, 1, 0};
    std::vector<int> V = {1, 2, 2};
    DRESS g(3, U, V);

    // High precision (small epsilon) but calculate NO histogram.
    // This previously would OOM. Now it might return empty map.
    auto r = g.deltaFit(0, 100, 1e-9, true, false);

    if (!r.histogram.empty()) {
        std::fprintf(stderr, "FAIL: histogram should be empty\n");
        g_fail++;
    } else {
        g_pass++;
    }

    if (r.multisets.size() != 3) {
        std::fprintf(stderr, "FAIL: multisets should have 3 entries, got %zu\n", r.multisets.size());
        g_fail++;
    } else {
        g_pass++;
    }
}

/* ── test: Δ^1 on K3 ──────────────────────────────────────────────── */

static void test_delta1_k3()
{
    std::printf("test_delta1_k3\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});
    auto r = g.deltaFit(1, 100, 1e-3);

    // C(3,1) = 3 subgraphs, each K2 with 1 edge
    ASSERT_EQ(hist_total(r.histogram), 3,
              "delta1 K3: 3 subgraphs * 1 edge = 3");
}

/* ── test: Δ^2 on K3 → zero edges ─────────────────────────────────── */

static void test_delta2_k3()
{
    std::printf("test_delta2_k3\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});
    auto r = g.deltaFit(2, 100, 1e-3);

    ASSERT_EQ(hist_total(r.histogram), 0, "delta2 K3: 0 edge values");
}

/* ── test: Δ^0 on K4 ──────────────────────────────────────────────── */

static void test_delta0_k4()
{
    std::printf("test_delta0_k4\n");

    DRESS g(4, {0, 0, 0, 1, 1, 2}, {1, 2, 3, 2, 3, 3});
    auto r = g.deltaFit(0, 100, 1e-3);

    ASSERT_EQ(hist_total(r.histogram), 6, "delta0 K4: 6 edge values");
    
    // K4 is perfectly symmetric -> all edges have same value.
    ASSERT_EQ((int)r.histogram.size(), 1, "delta0 K4: 1 unique value");
    ASSERT_EQ(r.histogram.front().second, 6, "count is 6");
}

/* ── test: Δ^1 on K4 ──────────────────────────────────────────────── */

static void test_delta1_k4()
{
    std::printf("test_delta1_k4\n");

    DRESS g(4, {0, 0, 0, 1, 1, 2}, {1, 2, 3, 2, 3, 3});
    auto r = g.deltaFit(1, 100, 1e-3);

    // C(4,1) = 4 subgraphs, each K3 with 3 edges
    ASSERT_EQ(hist_total(r.histogram), 12,
              "delta1 K4: 4 * 3 = 12 edge values");
    
    // All subgraphs are K3, all edges are 2.0 (same).
    ASSERT_EQ((int)r.histogram.size(), 1, "delta1 K4: 1 unique value");
    ASSERT_EQ(r.histogram.front().second, 12, "count is 12");
}

/* ── test: Δ^2 on K4 ──────────────────────────────────────────────── */

static void test_delta2_k4()
{
    std::printf("test_delta2_k4\n");

    DRESS g(4, {0, 0, 0, 1, 1, 2}, {1, 2, 3, 2, 3, 3});
    auto r = g.deltaFit(2, 100, 1e-3);

    // C(4,2) = 6 subgraphs, each K2 with 1 edge
    ASSERT_EQ(hist_total(r.histogram), 6,
              "delta2 K4: 6 * 1 = 6 edge values");
}

/* ── test: k >= N returns empty histogram ──────────────────────────── */

static void test_k_ge_N()
{
    std::printf("test_k_ge_N\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});

    auto r1 = g.deltaFit(3, 100, 1e-3);
    ASSERT_EQ(hist_total(r1.histogram), 0, "k == N: empty histogram");

    auto r2 = g.deltaFit(10, 100, 1e-3);
    ASSERT_EQ(hist_total(r2.histogram), 0, "k > N: empty histogram");
}

/* ── test: precompute flag ─────────────────────────────────────────── */

static void test_precompute()
{
    std::printf("test_precompute\n");

    /* precompute = false */
    DRESS g1(4, {0, 0, 0, 1, 1, 2}, {1, 2, 3, 2, 3, 3},
             DRESS_VARIANT_UNDIRECTED, false);
    /* precompute = true */
    DRESS g2(4, {0, 0, 0, 1, 1, 2}, {1, 2, 3, 2, 3, 3},
             DRESS_VARIANT_UNDIRECTED, true);

    auto r1 = g1.deltaFit(1, 100, 1e-3);
    auto r2 = g2.deltaFit(1, 100, 1e-3);

    ASSERT_EQ((int)r1.histogram.size(), (int)r2.histogram.size(), "precompute: same hist size");

    bool match = hist_equal(r1.histogram, r2.histogram);
    ASSERT(match, "precompute: identical histograms");
}

/* ── test: path P4 — edges not all equal ───────────────────────────── */

static void test_delta0_path()
{
    std::printf("test_delta0_path\n");

    DRESS g(4, {0, 1, 2}, {1, 2, 3});
    auto r = g.deltaFit(0, 100, 1e-3);

    ASSERT_EQ(hist_total(r.histogram), 3, "delta0 P4: 3 edges");

    // Edges are not all equal, so map size > 1.
    ASSERT_GT(r.histogram.size(), 1, "delta0 P4: edges not all equal");
}

/* ── test: multisets disabled ──────────────────────────────────────── */

static void test_multisets_disabled()
{
    std::printf("test_multisets_disabled\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});
    auto r = g.deltaFit(0, 100, 1e-3, false);

    ASSERT(r.multisets.empty(), "multisets should be empty");
}

/* ── test: multisets Δ^0 K3 ───────────────────────────────────────── */

static void test_multisets_delta0_k3()
{
    std::printf("test_multisets_delta0_k3\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});
    auto r = g.deltaFit(0, 100, 1e-3, true);

    ASSERT_EQ(r.num_subgraphs, int64_t(1), "C(3,0) = 1");
    ASSERT_EQ((int)r.multisets.size(), 3, "1 subgraph * 3 edges = 3");
    for (int i = 0; i < 3; i++)
        ASSERT(std::fabs(r.multisets[i] - 2.0) < 1e-3,
               "all K3 edges = 2.0");
}

/* ── test: multisets Δ^1 K3 (NaN pattern) ─────────────────────────── */

static void test_multisets_delta1_k3()
{
    std::printf("test_multisets_delta1_k3\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});
    auto r = g.deltaFit(1, 100, 1e-3, true);

    ASSERT_EQ(r.num_subgraphs, int64_t(3), "C(3,1) = 3");
    ASSERT_EQ((int)r.multisets.size(), 9, "3 subgraphs * 3 edges = 9");

    int E = 3;
    for (int s = 0; s < 3; s++) {
        int nans = 0;
        for (int e = 0; e < E; e++) {
            double v = r.multisets[s * E + e];
            if (std::isnan(v)) {
                nans++;
            } else {
                ASSERT(std::fabs(v - 2.0) < 1e-3, "non-NaN value = 2.0");
            }
        }
        ASSERT_EQ(nans, 2, "2 NaN per row");
    }
}

/* ── main ──────────────────────────────────────────────────────────── */

int main()
{
    test_hist_size();
    test_weighted_hist_size();
    test_delta0_k3();
    test_delta1_k3();
    test_delta2_k3();
    test_delta0_k4();
    test_delta1_k4();
    test_delta2_k4();
    test_k_ge_N();
    test_precompute();
    test_delta0_path();
    test_multisets_disabled();
    test_multisets_delta0_k3();
    test_multisets_delta1_k3();

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
