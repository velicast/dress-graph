#include "dress/delta_dress.h"

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
static p_dress_graph_t build_subgraph(p_dress_graph_t g,
                                      const int *del, int k)
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
            idx++;
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

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

int64_t *delta_fit(p_dress_graph_t g, int k, int iterations,
                   double epsilon, int precompute, int *hist_size)
{
    int N = g->N;
    int nbins = (int)(2.0 / epsilon) + 1;

    if (hist_size)
        *hist_size = nbins;

    int64_t *hist = (int64_t *)calloc(nbins, sizeof(int64_t));
    if (!hist) return NULL;

    /* ── k = 0: Δ^0 — run DRESS on the full graph ──────────────── */
    if (k == 0) {
        // Copy edge list (init_dress_graph takes ownership).
        int E = g->E;
        int *cp_U = (int *)malloc(E * sizeof(int));
        int *cp_V = (int *)malloc(E * sizeof(int));
        memcpy(cp_U, g->U, E * sizeof(int));
        memcpy(cp_V, g->V, E * sizeof(int));

        p_dress_graph_t sub = init_dress_graph(
            N, E, cp_U, cp_V, NULL, g->variant, precompute);

        fit(sub, iterations, epsilon, NULL, NULL);
        accumulate_histogram(sub, hist, nbins, epsilon);
        free_dress_graph(sub);
        return hist;
    }

    /* ── k >= N: no valid deletion subsets ───────────────────────── */
    if (k >= N) {
        return hist;   // all-zero histogram
    }

    /* ── k >= 1: iterative DFS over C(N, k) combinations ───────── */
    //
    // combo[0 .. k-1] holds the current k-subset in sorted order.
    // `depth` is the DFS stack pointer: combo[0..depth-1] are fixed,
    // and combo[depth] is the value being explored.
    //
    // Invariant:  combo[i] < combo[i+1]  for all valid i.
    //
    // At each step:
    //   1. Increment combo[depth].
    //   2. If combo[depth] exceeds its upper bound (N - k + depth),
    //      backtrack (depth--).
    //   3. If depth == k-1, we have a complete combination — process it.
    //   4. Otherwise, descend: depth++, seed combo[depth] from its
    //      predecessor (will be incremented at step 1).

    int *combo = (int *)malloc(k * sizeof(int));

    int depth = 0;
    combo[0] = -1;              // will be incremented to 0 on first iter

    while (depth >= 0) {
        combo[depth]++;

        // Upper bound for combo[depth]: ensure room for remaining slots.
        if (combo[depth] > N - k + depth) {
            depth--;            // backtrack
            continue;
        }

        if (depth == k - 1) {
            // ── complete k-subset: combo[0..k-1] ──
            p_dress_graph_t sub = build_subgraph(g, combo, k);
            if (sub) {
                fit(sub, iterations, epsilon, NULL, NULL);
                accumulate_histogram(sub, hist, nbins, epsilon);
                free_dress_graph(sub);
            }
        } else {
            // ── descend: seed next depth from current value ──
            depth++;
            combo[depth] = combo[depth - 1];  // incremented at top
        }
    }

    free(combo);
    return hist;
}
