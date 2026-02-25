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

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
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

static int64_t hist_total(const std::vector<int64_t> &h)
{
    return std::accumulate(h.begin(), h.end(), int64_t(0));
}

/* ── test: histogram size ──────────────────────────────────────────── */

static void test_hist_size()
{
    std::printf("test_hist_size\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});

    auto r1 = g.deltaFit(0, 100, 1e-3);
    ASSERT_EQ(r1.hist_size, 2001, "hist_size == 2001 for eps=1e-3");

    auto r2 = g.deltaFit(0, 100, 1e-6);
    ASSERT_EQ(r2.hist_size, 2000001, "hist_size == 2000001 for eps=1e-6");
}

/* ── test: Δ^0 on K3 ──────────────────────────────────────────────── */

static void test_delta0_k3()
{
    std::printf("test_delta0_k3\n");

    DRESS g(3, {0, 1, 0}, {1, 2, 2});
    auto r = g.deltaFit(0, 100, 1e-3);

    ASSERT_EQ(hist_total(r.histogram), 3, "delta0 K3: 3 edge values");

    // K3: all edges equal → single non-zero bin
    int nonzero = 0;
    for (auto v : r.histogram)
        if (v > 0) nonzero++;
    ASSERT_EQ(nonzero, 1, "delta0 K3: all edges in same bin");

    // Top bin holds value 2.0
    ASSERT_GT(r.histogram[r.hist_size - 1], 0,
              "delta0 K3: top bin holds value 2.0");
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
    ASSERT_EQ(r.histogram[r.hist_size - 1], 6,
              "delta0 K4: all edges at top bin");
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
    ASSERT_EQ(r.histogram[r.hist_size - 1], 12,
              "delta1 K4: all 12 edges at top bin");
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

    DRESS g(4, {0, 0, 0, 1, 1, 2}, {1, 2, 3, 2, 3, 3});

    auto r1 = g.deltaFit(1, 100, 1e-3, /*precompute=*/false);
    auto r2 = g.deltaFit(1, 100, 1e-3, /*precompute=*/true);

    ASSERT_EQ(r1.hist_size, r2.hist_size, "precompute: same hist_size");

    bool match = true;
    for (int i = 0; i < r1.hist_size; i++) {
        if (r1.histogram[i] != r2.histogram[i]) { match = false; break; }
    }
    ASSERT(match, "precompute: identical histograms");
}

/* ── test: path P4 — edges not all equal ───────────────────────────── */

static void test_delta0_path()
{
    std::printf("test_delta0_path\n");

    DRESS g(4, {0, 1, 2}, {1, 2, 3});
    auto r = g.deltaFit(0, 100, 1e-3);

    ASSERT_EQ(hist_total(r.histogram), 3, "delta0 P4: 3 edges");

    int nonzero = 0;
    for (auto v : r.histogram)
        if (v > 0) nonzero++;
    ASSERT_GT(nonzero, 1, "delta0 P4: edges not all equal");
}

/* ── main ──────────────────────────────────────────────────────────── */

int main()
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

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
