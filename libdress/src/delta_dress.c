#include "dress/delta_dress.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                   */
/* ------------------------------------------------------------------ */

// Build a subgraph by removing the vertices listed in `del` (sorted,
// length `k`) from the original graph `g`.
//
// Returns a freshly allocated dress graph with remapped vertex ids
// (0 .. N-k-1) and only the edges whose both endpoints survive.
// Returns NULL if the subgraph has zero edges.
//
// If `edge_map` is non-NULL (size E), fills edge_map[e] with the
// subgraph edge index of original edge e, or -1 if edge e was removed.
static p_dress_graph_t build_subgraph(p_dress_graph_t g,
                                      const int *del, int k,
                                      int *edge_map)
{
    int N = g->N;
    int E = g->E;
    int sub_N = N - k;

    // Map old vertex id -> new id (-1 = deleted).
    // Since `del` is sorted we can scan both arrays in one pass.
    int *node_map = (int *)malloc(N * sizeof(int));
    int di = 0;   // index into del[]
    int new_id = 0;
    for (int v = 0; v < N; v++) {
        if (di < k && v == del[di]) {
            node_map[v] = -1;
            di++;
        } else {
            node_map[v] = new_id++;
        }
    }

    // Count surviving edges.
    int sub_E = 0;
    for (int e = 0; e < E; e++) {
        if (node_map[g->U[e]] >= 0 && node_map[g->V[e]] >= 0)
            sub_E++;
    }

    free(node_map);

    if (sub_E == 0) {
        // Fill edge_map with -1 if requested.
        if (edge_map) {
            for (int e = 0; e < E; e++) edge_map[e] = -1;
        }
        return NULL;
    }

    // Rebuild the mapping (we freed it above to keep the hot path lean;
    // the two-pass approach avoids an extra malloc for sub_E counts).
    node_map = (int *)malloc(N * sizeof(int));
    di = 0;
    new_id = 0;
    for (int v = 0; v < N; v++) {
        if (di < k && v == del[di]) {
            node_map[v] = -1;
            di++;
        } else {
            node_map[v] = new_id++;
        }
    }

    // Allocate edge arrays (init_dress_graph takes ownership).
    int *sub_U = (int *)malloc(sub_E * sizeof(int));
    int *sub_V = (int *)malloc(sub_E * sizeof(int));
    int idx = 0;
    for (int e = 0; e < E; e++) {
        int mu = node_map[g->U[e]];
        int mv = node_map[g->V[e]];
        if (mu >= 0 && mv >= 0) {
            sub_U[idx] = mu;
            sub_V[idx] = mv;
            if (edge_map) edge_map[e] = idx;
            idx++;
        } else {
            if (edge_map) edge_map[e] = -1;
        }
    }

    free(node_map);

    return init_dress_graph(sub_N, sub_E, sub_U, sub_V,
                            NULL, g->variant,
                            g->precompute_intercepts);
}

// Accumulate converged edge dress values from `sub` into `hist`.
//   bin(d) = clamp( floor(d / epsilon), 0, nbins-1 )
static void accumulate_histogram(p_dress_graph_t sub,
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

// Fill one row of the multisets matrix.
//   multisets[row * orig_E + e] = sub->edge_dress[ edge_map[e] ]
//   or NAN when edge_map[e] == -1.
static void fill_multiset_row(p_dress_graph_t sub,
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

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

/* Compute C(n, k) — binomial coefficient. */
static int64_t binom(int n, int k)
{
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;   /* C(n,k) = C(n, n-k) */
    int64_t r = 1;
    for (int i = 0; i < k; i++) {
        r = r * (n - i) / (i + 1);
    }
    return r;
}

int64_t *delta_fit(p_dress_graph_t g, int k, int iterations,
                   double epsilon, int *hist_size,
                   int keep_multisets, double **multisets,
                   int64_t *num_subgraphs)
{
    int N = g->N;
    int E = g->E;
    int nbins = (int)(2.0 / epsilon) + 1;

    if (hist_size)
        *hist_size = nbins;

    int64_t *hist = (int64_t *)calloc(nbins, sizeof(int64_t));
    if (!hist) return NULL;

    /* Compute C(N, k) and optionally allocate multisets buffer. */
    int64_t cnk = (k == 0) ? 1 : binom(N, k);
    if (num_subgraphs) *num_subgraphs = cnk;

    int wants_ms = keep_multisets && multisets;
    double *ms = NULL;
    if (wants_ms) {
        ms = (double *)malloc((size_t)cnk * E * sizeof(double));
        *multisets = ms;
        if (!ms) wants_ms = 0;  /* allocation failed — fall back */
    }

    /* ── k = 0: Δ^0 — run DRESS on the full graph ──────────────── */
    if (k == 0) {
        // Copy edge list (init_dress_graph takes ownership).
        int *cp_U = (int *)malloc(E * sizeof(int));
        int *cp_V = (int *)malloc(E * sizeof(int));
        memcpy(cp_U, g->U, E * sizeof(int));
        memcpy(cp_V, g->V, E * sizeof(int));

        p_dress_graph_t sub = init_dress_graph(
            N, E, cp_U, cp_V, NULL, g->variant, g->precompute_intercepts);

        fit(sub, iterations, epsilon, NULL, NULL);
        accumulate_histogram(sub, hist, nbins, epsilon);

        if (wants_ms) {
            // k=0: single subgraph (s=0), all edges survive, identity map.
            for (int e = 0; e < E; e++)
                ms[e] = sub->edge_dress[e];
        }

        free_dress_graph(sub);
        return hist;
    }

    /* ── k >= N: no valid deletion subsets ───────────────────────── */
    if (k >= N) {
        return hist;   // all-zero histogram
    }

    /* ── k >= 1: iterative DFS over C(N, k) combinations ───────── */

    int *combo = (int *)malloc(k * sizeof(int));
    int *edge_map = wants_ms ? (int *)malloc(E * sizeof(int)) : NULL;

    int depth = 0;
    combo[0] = -1;              // will be incremented to 0 on first iter
    int64_t s = 0;              // subgraph counter for multisets rows

    while (depth >= 0) {
        combo[depth]++;

        // Upper bound for combo[depth]: ensure room for remaining slots.
        if (combo[depth] > N - k + depth) {
            depth--;            // backtrack
            continue;
        }

        if (depth == k - 1) {
            // ── complete k-subset: combo[0..k-1] ──
            p_dress_graph_t sub = build_subgraph(g, combo, k, edge_map);
            if (sub) {
                fit(sub, iterations, epsilon, NULL, NULL);
                accumulate_histogram(sub, hist, nbins, epsilon);
                if (wants_ms)
                    fill_multiset_row(sub, edge_map, E,
                                      ms + s * E);
                free_dress_graph(sub);
            } else if (wants_ms) {
                // Zero-edge subgraph: fill row with NAN.
                double *row = ms + s * E;
                for (int e = 0; e < E; e++) row[e] = NAN;
            }
            s++;
        } else {
            // ── descend: seed next depth from current value ──
            depth++;
            combo[depth] = combo[depth - 1];  // incremented at top
        }
    }

    free(combo);
    free(edge_map);
    return hist;
}
