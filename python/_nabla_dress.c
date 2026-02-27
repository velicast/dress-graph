#include "dress/nabla_dress.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                   */
/* ------------------------------------------------------------------ */

// Accumulate converged edge dress values into `hist`.
//   bin(d) = clamp( floor(d / epsilon), 0, nbins-1 )
static void nabla_accumulate_histogram(p_dress_graph_t g,
                                 int64_t *hist, int nbins,
                                 double epsilon)
{
    for (int e = 0; e < g->E; e++) {
        int bin = (int)(g->edge_dress[e] / epsilon);
        if (bin < 0)      bin = 0;
        if (bin >= nbins)  bin = nbins - 1;
        hist[bin]++;
    }
}

// Reset edge_dress and edge_dress_next to 1.0 for all edges.
static void reset_dress_values(p_dress_graph_t g)
{
    for (int e = 0; e < g->E; e++) {
        g->edge_dress[e]      = 1.0;
        g->edge_dress_next[e] = 1.0;
    }
}

// Apply individualization weights for a k-subset `combo[0..k-1]`.
// Restores original weights first, then scales every edge incident to
// any vertex in the combo by nabla_weight (multiplicative).
static void apply_nabla_weights(p_dress_graph_t g,
                               const double *orig_weights,
                               const int *combo, int k,
                               double nabla_weight,
                               const int *inci_offset,
                               const int *inci_edges)
{
    int E = g->E;
    memcpy(g->edge_weight, orig_weights, E * sizeof(double));
    for (int j = 0; j < k; j++) {
        int v = combo[j];
        for (int i = inci_offset[v]; i < inci_offset[v + 1]; i++) {
            g->edge_weight[inci_edges[i]] *= nabla_weight;
        }
    }
}

/* Compute C(n, k) — binomial coefficient. */
static int64_t nabla_binom(int n, int k)
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

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

int64_t *nabla_fit(p_dress_graph_t g, int k, int iterations,
                  double epsilon, double nabla_weight,
                  int *hist_size,
                  int keep_multisets, double **multisets,
                  int64_t *num_subsets)
{
    int N = g->N;
    int E = g->E;
    int nbins = (int)(2.0 / epsilon) + 1;

    if (hist_size)
        *hist_size = nbins;

    int64_t *hist = (int64_t *)calloc(nbins, sizeof(int64_t));
    if (!hist) return NULL;

    /* Compute C(N, k) and optionally allocate multisets buffer. */
    int64_t cnk = (k == 0) ? 1 : nabla_binom(N, k);
    if (num_subsets) *num_subsets = cnk;

    int wants_ms = keep_multisets && multisets;
    double *ms = NULL;
    if (wants_ms) {
        ms = (double *)malloc((size_t)cnk * E * sizeof(double));
        *multisets = ms;
        if (!ms) wants_ms = 0;
    }

    /* Save original edge weights so we can restore after each round. */
    double *orig_weights = (double *)malloc(E * sizeof(double));
    memcpy(orig_weights, g->edge_weight, E * sizeof(double));

    /* Build incidence index: for each vertex v, which edge indices
       touch v?  Stored in a flat CSR-like layout.
       inci_offset[v] .. inci_offset[v+1] index into inci_edges[]. */
    int *inci_count = (int *)calloc(N, sizeof(int));
    for (int e = 0; e < E; e++) {
        inci_count[g->U[e]]++;
        inci_count[g->V[e]]++;
    }
    int *inci_offset = (int *)malloc((N + 1) * sizeof(int));
    inci_offset[0] = 0;
    for (int v = 0; v < N; v++)
        inci_offset[v + 1] = inci_offset[v] + inci_count[v];

    int *inci_edges = (int *)malloc(inci_offset[N] * sizeof(int));
    memset(inci_count, 0, N * sizeof(int));
    for (int e = 0; e < E; e++) {
        int u = g->U[e], v = g->V[e];
        inci_edges[inci_offset[u] + inci_count[u]++] = e;
        inci_edges[inci_offset[v] + inci_count[v]++] = e;
    }
    free(inci_count);

    /* ── k = 0: Nabla^0 — run DRESS on the unmodified graph ─────── */
    if (k == 0) {
        reset_dress_values(g);
        fit(g, iterations, epsilon, NULL, NULL);
        nabla_accumulate_histogram(g, hist, nbins, epsilon);

        if (wants_ms) {
            for (int e = 0; e < E; e++)
                ms[e] = g->edge_dress[e];
        }

        reset_dress_values(g);
        free(orig_weights);
        free(inci_offset);
        free(inci_edges);
        return hist;
    }

    /* ── k >= N: no valid subsets ───────────────────────────────── */
    if (k >= N) {
        free(orig_weights);
        free(inci_offset);
        free(inci_edges);
        return hist;
    }

    /* ── k >= 1: iterative DFS over C(N, k) combinations ───────── */

    int *combo = (int *)malloc(k * sizeof(int));
    int depth = 0;
    combo[0] = -1;              // will be incremented to 0 on first iter
    int64_t s = 0;              // subset counter for multisets rows

    while (depth >= 0) {
        combo[depth]++;

        // Upper bound for combo[depth]: ensure room for remaining slots.
        if (combo[depth] > N - k + depth) {
            depth--;            // backtrack
            continue;
        }

        if (depth == k - 1) {
            // ── complete k-subset: combo[0..k-1] ──

            /* 1. Reset dress values to 1.0. */
            reset_dress_values(g);

            /* 2. Restore original weights, then anchor incident edges. */
            apply_nabla_weights(g, orig_weights, combo, k,
                               nabla_weight, inci_offset, inci_edges);

            /* 3. Run DRESS fitting. */
            fit(g, iterations, epsilon, NULL, NULL);

            /* 4. Accumulate histogram. */
            nabla_accumulate_histogram(g, hist, nbins, epsilon);

            /* 5. Optionally store multiset row. */
            if (wants_ms) {
                double *row = ms + s * E;
                for (int e = 0; e < E; e++)
                    row[e] = g->edge_dress[e];
            }
            s++;
        } else {
            // ── descend: seed next depth from current value ──
            depth++;
            combo[depth] = combo[depth - 1];  // incremented at top
        }
    }

    /* Restore original weights and dress values. */
    memcpy(g->edge_weight, orig_weights, E * sizeof(double));
    reset_dress_values(g);

    free(combo);
    free(orig_weights);
    free(inci_offset);
    free(inci_edges);

    return hist;
}
