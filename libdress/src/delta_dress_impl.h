/**
 * delta_dress_impl.h — shared implementation for Δ^k-DRESS.
 *
 * Internal header included by delta_dress.c and cuda/delta_dress_cuda.c.
 * Parameterised by a function pointer so the same combination-enumeration
 * and histogram logic drives both the CPU and CUDA backends.
 */

#ifndef DELTA_DRESS_IMPL_H
#define DELTA_DRESS_IMPL_H

#include "dress/dress.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Fit-function signature: same as dress_fit / dress_fit_cuda. */
typedef void (*dress_fit_fn)(p_dress_graph_t, int, double, int *, double *);

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                   */
/* ------------------------------------------------------------------ */

/* Build a subgraph by removing the vertices listed in `del` (sorted,
 * length `k`) from the original graph `g`.
 *
 * Returns a freshly allocated dress graph with remapped vertex ids
 * (0 .. N-k-1) and only the edges whose both endpoints survive.
 * Returns NULL if the subgraph has zero edges.
 *
 * If `edge_map` is non-NULL (size E), fills edge_map[e] with the
 * subgraph edge index of original edge e, or -1 if edge e was removed. */
static p_dress_graph_t delta_build_subgraph(p_dress_graph_t g,
                                            const int *del, int k,
                                            int *edge_map)
{
    int N = g->N;
    int E = g->E;
    int sub_N = N - k;

    /* Map old vertex id -> new id (-1 = deleted). */
    int *node_map = (int *)malloc(N * sizeof(int));
    int di = 0, new_id = 0;
    for (int v = 0; v < N; v++) {
        if (di < k && v == del[di]) {
            node_map[v] = -1;
            di++;
        } else {
            node_map[v] = new_id++;
        }
    }

    /* Count surviving edges. */
    int sub_E = 0;
    for (int e = 0; e < E; e++) {
        if (node_map[g->U[e]] >= 0 && node_map[g->V[e]] >= 0)
            sub_E++;
    }

    free(node_map);

    if (sub_E == 0) {
        if (edge_map) {
            for (int e = 0; e < E; e++) edge_map[e] = -1;
        }
        return NULL;
    }

    /* Rebuild mapping (two-pass keeps hot path lean). */
    node_map = (int *)malloc(N * sizeof(int));
    di = 0; new_id = 0;
    for (int v = 0; v < N; v++) {
        if (di < k && v == del[di]) {
            node_map[v] = -1;
            di++;
        } else {
            node_map[v] = new_id++;
        }
    }

    /* Allocate edge arrays (init_dress_graph takes ownership). */
    int *sub_U = (int *)malloc(sub_E * sizeof(int));
    int *sub_V = (int *)malloc(sub_E * sizeof(int));
    double *sub_W = NULL;
    if (g->W != NULL)
        sub_W = (double *)malloc(sub_E * sizeof(double));
    int idx = 0;
    for (int e = 0; e < E; e++) {
        int mu = node_map[g->U[e]];
        int mv = node_map[g->V[e]];
        if (mu >= 0 && mv >= 0) {
            sub_U[idx] = mu;
            sub_V[idx] = mv;
            if (sub_W) sub_W[idx] = g->W[e];
            if (edge_map) edge_map[e] = idx;
            idx++;
        } else {
            if (edge_map) edge_map[e] = -1;
        }
    }

    free(node_map);

    return init_dress_graph(sub_N, sub_E, sub_U, sub_V,
                            sub_W, g->variant,
                            g->precompute_intercepts);
}

/* Accumulate converged edge dress values into histogram. */
static void delta_accumulate_histogram(p_dress_graph_t sub,
                                       int64_t *hist, int nbins,
                                       double epsilon)
{
    for (int e = 0; e < sub->E; e++) {
        int bin = (int)(sub->edge_dress[e] / epsilon);
        if (bin < 0)      bin = 0;
        if (bin >= nbins)  bin = nbins - 1;
        hist[bin]++;
    }
}

/* Fill one row of the multisets matrix. */
static void delta_fill_multiset_row(p_dress_graph_t sub,
                                    const int *edge_map, int orig_E,
                                    double *row)
{
    for (int e = 0; e < orig_E; e++) {
        if (edge_map[e] >= 0)
            row[e] = sub->edge_dress[edge_map[e]];
        else
            row[e] = NAN;
    }
}

/* Binomial coefficient C(n, k). */
static int64_t delta_binom(int n, int k)
{
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;
    int64_t r = 1;
    for (int i = 0; i < k; i++) {
        r = r * (n - i) / (i + 1);
    }
    return r;
}

/* Upper bound on maximum DRESS value.
 * Unweighted: exactly 2.0.
 * Weighted: solve d = r(d) + 1/r(d) by fixed-point iteration. */
static double delta_compute_dmax_bound(p_dress_graph_t g)
{
    if (g->W == NULL)
        return 2.0;

    double Smin = 1e308, Smax = 0.0;
    for (int u = 0; u < g->N; u++) {
        double s = 0.0;
        int base = g->adj_offset[u];
        int end  = g->adj_offset[u + 1];
        for (int i = base; i < end; i++) {
            int ei = g->adj_edge_idx[i];
            s += g->edge_weight[ei];
        }
        if (s < Smin) Smin = s;
        if (s > Smax) Smax = s;
    }

    if (Smin <= 0.0 || Smax == Smin)
        return 2.0;

    double d = 2.0;
    for (int i = 0; i < 50; i++) {
        double r = sqrt((4.0 + Smax * d) / (4.0 + Smin * d));
        double d_new = r + 1.0 / r;
        if (fabs(d_new - d) < 1e-12) break;
        d = d_new;
    }
    return d;
}

/* ------------------------------------------------------------------ */
/*  Shared Δ^k-DRESS implementation                                    */
/* ------------------------------------------------------------------ */

/**
 * Core Δ^k-DRESS: enumerate C(N,k) deletion subsets, fit each subgraph
 * using `fit_fn`, and accumulate the pooled histogram.
 */
static int64_t *delta_dress_fit_impl(p_dress_graph_t g, int k,
                                     int iterations, double epsilon,
                                     int *hist_size,
                                     int keep_multisets,
                                     double **multisets,
                                     int64_t *num_subgraphs,
                                     dress_fit_fn fit_fn)
{
    int N = g->N;
    int E = g->E;
    double dmax = delta_compute_dmax_bound(g);
    int nbins = (int)(dmax / epsilon) + 1;

    if (hist_size)
        *hist_size = nbins;

    int64_t *hist = (int64_t *)calloc(nbins, sizeof(int64_t));
    if (!hist) return NULL;

    int64_t cnk = (k == 0) ? 1 : delta_binom(N, k);
    if (num_subgraphs) *num_subgraphs = cnk;

    int wants_ms = keep_multisets && multisets;
    double *ms = NULL;
    if (wants_ms) {
        ms = (double *)malloc((size_t)cnk * E * sizeof(double));
        *multisets = ms;
        if (!ms) wants_ms = 0;
    }

    /* ── k = 0: Δ^0 — run DRESS on the full graph ──────────────── */
    if (k == 0) {
        int *cp_U = (int *)malloc(E * sizeof(int));
        int *cp_V = (int *)malloc(E * sizeof(int));
        double *cp_W = NULL;
        memcpy(cp_U, g->U, E * sizeof(int));
        memcpy(cp_V, g->V, E * sizeof(int));
        if (g->W != NULL) {
            cp_W = (double *)malloc(E * sizeof(double));
            memcpy(cp_W, g->W, E * sizeof(double));
        }

        p_dress_graph_t sub = init_dress_graph(
            N, E, cp_U, cp_V, cp_W, g->variant, g->precompute_intercepts);

        fit_fn(sub, iterations, epsilon, NULL, NULL);
        delta_accumulate_histogram(sub, hist, nbins, epsilon);

        if (wants_ms) {
            for (int e = 0; e < E; e++)
                ms[e] = sub->edge_dress[e];
        }

        free_dress_graph(sub);
        return hist;
    }

    /* ── k >= N: no valid deletion subsets ───────────────────────── */
    if (k >= N) {
        return hist;
    }

    /* ── k >= 1: iterative DFS over C(N, k) combinations ───────── */

    int *combo = (int *)malloc(k * sizeof(int));
    int *edge_map = wants_ms ? (int *)malloc(E * sizeof(int)) : NULL;

    int depth = 0;
    combo[0] = -1;
    int64_t s = 0;

    while (depth >= 0) {
        combo[depth]++;

        if (combo[depth] > N - k + depth) {
            depth--;
            continue;
        }

        if (depth == k - 1) {
            p_dress_graph_t sub = delta_build_subgraph(g, combo, k, edge_map);
            if (sub) {
                fit_fn(sub, iterations, epsilon, NULL, NULL);
                delta_accumulate_histogram(sub, hist, nbins, epsilon);
                if (wants_ms)
                    delta_fill_multiset_row(sub, edge_map, E,
                                            ms + s * E);
                free_dress_graph(sub);
            } else if (wants_ms) {
                double *row = ms + s * E;
                for (int e = 0; e < E; e++) row[e] = NAN;
            }
            s++;
        } else {
            depth++;
            combo[depth] = combo[depth - 1];
        }
    }

    free(combo);
    free(edge_map);
    return hist;
}

#endif /* DELTA_DRESS_IMPL_H */
