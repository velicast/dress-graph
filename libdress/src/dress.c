#include "dress/dress.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Internal types                                                     */
/* ------------------------------------------------------------------ */

// Temporary adjacency entry used during graph construction.
// Carries the neighbor id, original edge index, and edge weight.
typedef struct __adj_edge_t {
    int    x;      // neighbor vertex id
    int    e_idx;  // index into the input edge list
    double w;      // edge weight (possibly doubled for undirected)
} adj_edge_t, *p_adj_edge_t;

/* ------------------------------------------------------------------ */
/*  Static helpers                                                     */
/* ------------------------------------------------------------------ */

// Comparator for sorting adj_edge_t entries by neighbor id (ascending).
static int edge_cmp(const void *a, const void *b)
{
    const adj_edge_t *ea = (const adj_edge_t *)a;
    const adj_edge_t *eb = (const adj_edge_t *)b;
    return (ea->x > eb->x) - (ea->x < eb->x);
}

// Find edge u->v in raw adjacency (u's segment is sorted by neighbor id).
// Returns index in raw_data if found, -1 otherwise.
static int find_raw_edge(int u, int v, const int *raw_offset, const adj_edge_t *raw_data)
{
    int lo = raw_offset[u];
    int hi = raw_offset[u + 1] - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int x = raw_data[mid].x;
        if (x == v) {
            return mid;
        }
        if (x < v) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/*  Graph construction                                                 */
/* ------------------------------------------------------------------ */

// Build the raw (input-level) adjacency in a flat CSR-like layout.
//
// For UNDIRECTED, each edge (u,v) produces two entries: v in N(u) and u in N(v).
// For directed variants, only the out-edge u→v is stored.
//
// Output:
//   *out_offset : [N+1] prefix-sum array of per-node degrees
//   *out_data   : flat array of adj_edge_t, sorted by neighbor id per node
//
// Both output arrays are heap-allocated; caller takes ownership.
static void build_raw_adjacency(p_dress_graph_t g,
                                int **out_offset, adj_edge_t **out_data,
                                double *W)
{
    int N = g->N, E = g->E, i;
    dress_variant_t variant = g->variant;
    int *U = g->U, *V = g->V;
    int *cnt = (int *)calloc(N, sizeof(int));

    // Count per-node degree
    for (i = 0; i < E; i++) {
        int u = U[i], v = V[i];
        if (variant == DRESS_VARIANT_UNDIRECTED) { cnt[u]++; cnt[v]++; }
        else                                      { cnt[u]++; }
    }

    // Compute CSR offsets via prefix sum
    int *offset = (int *)malloc((N + 1) * sizeof(int));
    offset[0] = 0;
    for (i = 0; i < N; i++)
        offset[i + 1] = offset[i] + cnt[i];

    // Single flat allocation for all adjacency entries
    adj_edge_t *data = (adj_edge_t *)malloc(offset[N] * sizeof(adj_edge_t));

    // Scatter edges into their respective node segments
    memset(cnt, 0, N * sizeof(int));
    for (i = 0; i < E; i++) {
        int    u = U[i], v = V[i];
        double w = (W == NULL) ? 1.0 : W[i];

        if (variant == DRESS_VARIANT_UNDIRECTED) {
            int pu = offset[u] + cnt[u]++;
            int pv = offset[v] + cnt[v]++;
            data[pu].x = v;  data[pu].e_idx = i;  data[pu].w = w;
            data[pv].x = u;  data[pv].e_idx = i;  data[pv].w = w;
        } else {
            int pu = offset[u] + cnt[u]++;
            data[pu].x = v;  data[pu].e_idx = i;  data[pu].w = w;
        }
    }
    free(cnt);
    if (W != NULL) {
        free(W); // W is no longer needed after edge weights are copied into data
    }
    
    // Sort each node's segment by neighbor id for binary-search access
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (i = 0; i < N; i++) {
        int sz = offset[i + 1] - offset[i];
        if (sz > 1)
            qsort(&data[offset[i]], sz, sizeof(adj_edge_t), edge_cmp);
    }

    *out_offset = offset;
    *out_data   = data;
}

// Build the variant-specific adjacency and write it into the graph's
// final CSR arrays (adj_offset, adj_target, adj_edge_idx, edge_weight).
//
// The variant adjacency differs from the raw adjacency:
//   UNDIRECTED:  N[u] = N(u), weight doubled (w_uv = w_in + w_out = 2w)
//   DIRECTED:    N[u] = in(u) ∪ out(u), weight = w(u→v) + w(v→u) if both exist
//   FORWARD:     N[u] = out(u)
//   BACKWARD:    N[u] = in(u)
//
// Takes ownership of raw_offset and raw_data (freed internally).
static void build_variant_adjacency(p_dress_graph_t g,
                                    int *raw_offset, adj_edge_t *raw_data)
{
    int N = g->N, E = g->E, i, e, S;
    dress_variant_t variant = g->variant;
    int *U = g->U, *V = g->V;

    // Per-node degree counters for the variant adjacency.
    // For DIRECTED, N[u] = out(u) U in(u) with one entry per neighbor.
    int *node_out_degree = (int *)calloc(N + 1, sizeof(int));
    int *node_in_degree  = (int *)calloc(N + 1, sizeof(int));

    // Count variant-specific per-node degrees
    for (e = 0; e < E; e++) {
        int u = U[e], v = V[e];
        if (variant == DRESS_VARIANT_UNDIRECTED) {
            node_out_degree[u]++;
            node_out_degree[v]++;
        } else if (variant == DRESS_VARIANT_DIRECTED) {
            // Out-neighbor for u always appears in N[u].
            node_out_degree[u]++;

            // Add incoming neighbor for v only if reciprocal v->u is absent.
            // If v->u exists, u is already an out-neighbor of v.
            if (find_raw_edge(v, u, raw_offset, raw_data) < 0) {
                node_in_degree[v]++;
            }
        } else if (variant == DRESS_VARIANT_FORWARD) {
            node_out_degree[u]++;
        } else if (variant == DRESS_VARIANT_BACKWARD) {
            node_out_degree[v]++;
        }
    }

    // Compute CSR offsets for the temporary variant adjacency
    int *tmp_offset = (int *)malloc((N + 1) * sizeof(int));
    tmp_offset[0] = 0;
    for (i = 0; i < N; i++) {
        if (variant == DRESS_VARIANT_DIRECTED) {
            // Merge in-degree into out-degree to get total adjacency size
            node_out_degree[i] = node_out_degree[i] + node_in_degree[i];
        }
        tmp_offset[i + 1] = tmp_offset[i] + node_out_degree[i];
    }
    S = tmp_offset[N];

    free(node_out_degree);
    free(node_in_degree);
    adj_edge_t *tmp_data = (adj_edge_t *)malloc(S * sizeof(adj_edge_t));
    int *tmp_count = (int *)calloc(N, sizeof(int));

    // Populate the variant adjacency by iterating over each node's raw
    // neighbors and applying variant-specific rules.
    for (int u = 0; u < N; u++) {
        int raw_start = raw_offset[u];
        int raw_end   = raw_offset[u + 1];
        for (i = raw_start; i < raw_end; i++) {
            int    v   = raw_data[i].x;
            int    eid = raw_data[i].e_idx;
            double w   = raw_data[i].w;

            if (variant == DRESS_VARIANT_UNDIRECTED) {
                int pu = tmp_offset[u] + tmp_count[u]++;
                tmp_data[pu].x = v;  tmp_data[pu].e_idx = eid;  tmp_data[pu].w = 2.0 * w;
            } else if (variant == DRESS_VARIANT_DIRECTED) {
                // Directed: outgoing neighbor always in N[u].
                // Its weight is w(u->v) + w(v->u) when reciprocal exists.
                int reciprocal = find_raw_edge(v, u, raw_offset, raw_data);

                int pu = tmp_offset[u] + tmp_count[u]++;
                tmp_data[pu].x = v;  tmp_data[pu].e_idx = eid;  tmp_data[pu].w = w;
                if (reciprocal >= 0) {
                    tmp_data[pu].w += raw_data[reciprocal].w;
                }

                // Incoming-only neighbor for v if v has no out-edge to u.
                if (reciprocal < 0) {
                    int pv = tmp_offset[v] + tmp_count[v]++;
                    tmp_data[pv].x = u;  tmp_data[pv].e_idx = eid;  tmp_data[pv].w = w;
                }
            } else if (variant == DRESS_VARIANT_FORWARD) {
                // Forward: only outgoing edges
                int pu = tmp_offset[u] + tmp_count[u]++;
                tmp_data[pu].x = v;  tmp_data[pu].e_idx = eid;  tmp_data[pu].w = w;
            } else if (variant == DRESS_VARIANT_BACKWARD) {
                // Backward: only incoming edges
                int pv = tmp_offset[v] + tmp_count[v]++;
                tmp_data[pv].x = u;  tmp_data[pv].e_idx = eid;  tmp_data[pv].w = w;
            }
        }
    }

    // Raw adjacency no longer needed
    free(raw_offset);
    free(raw_data);
    free(tmp_count);

    // Transfer tmp_offset as the final CSR offset array (same layout)
    g->adj_offset   = tmp_offset;
    g->adj_target   = (int *)malloc(S * sizeof(int));
    g->adj_edge_idx = (int *)malloc(S * sizeof(int));
    g->edge_weight  = (double *)malloc(E * sizeof(double));

    // Sort each node's segment by neighbor id, then split the adj_edge_t
    // struct-of-arrays into the final CSR arrays
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (i = 0; i < N; i++) {
        int base = tmp_offset[i];
        int sz   = tmp_offset[i + 1] - tmp_offset[i];
        if (sz > 1)
            qsort(&tmp_data[base], sz, sizeof(adj_edge_t), edge_cmp);
        for (int j = 0; j < sz; j++) {
            g->adj_target[base + j]   = tmp_data[base + j].x;
            g->adj_edge_idx[base + j] = tmp_data[base + j].e_idx;
            g->edge_weight[tmp_data[base + j].e_idx] = tmp_data[base + j].w;
        }
    }

    free(tmp_data);
}

// Precompute neighborhood intercepts for every edge.
//
// For each edge e = (u,v), finds the set of common neighbors
// X = N[u] ∩ N[v] using a sorted-merge walk over the CSR adjacency.
// Stores the edge indices for (u,x) and (v,x) for each x ∈ X in flat
// arrays indexed by intercept_offset[e] .. intercept_offset[e+1].
//
// This trades O(∑|N[u]∩N[v]|) extra memory for reducing the per-edge
// iteration cost from O(deg_u + deg_v) to O(|N[u] ∩ N[v]|).
static void compute_intercepts(p_dress_graph_t g)
{
    int E = g->E, e, T;
    int *U = g->U, *V = g->V;

    g->intercept_offset = (int *)malloc((E + 1) * sizeof(int));

    // First pass: count |N[u] ∩ N[v]| for each edge
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (e = 0; e < E; e++) {
        int u  = U[e], v = V[e];
        int iu = g->adj_offset[u], iu_end = g->adj_offset[u + 1];
        int iv = g->adj_offset[v], iv_end = g->adj_offset[v + 1];
        int cnt = 0;

        while (iu < iu_end && iv < iv_end) {
            int x = g->adj_target[iu], y = g->adj_target[iv];
            if      (x == y) { cnt++; iu++; iv++; }
            else if (x < y)  { iu++; }
            else              { iv++; }
        }
        g->intercept_offset[e + 1] = cnt;
    }

    // Convert per-edge counts into a prefix sum (CSR offsets)
    g->intercept_offset[0] = 0;
    for (e = 0; e < E; e++)
        g->intercept_offset[e + 1] += g->intercept_offset[e];

    T = g->intercept_offset[E];
    g->intercept_edge_ux = (int *)malloc(T * sizeof(int));
    g->intercept_edge_vx = (int *)malloc(T * sizeof(int));

    // Second pass: record edge indices for each common neighbor
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (e = 0; e < E; e++) {
        int u  = U[e], v = V[e];
        int iu = g->adj_offset[u], iu_end = g->adj_offset[u + 1];
        int iv = g->adj_offset[v], iv_end = g->adj_offset[v + 1];
        int off = g->intercept_offset[e];

        while (iu < iu_end && iv < iv_end) {
            int x = g->adj_target[iu], y = g->adj_target[iv];
            if (x == y) {
                g->intercept_edge_ux[off] = g->adj_edge_idx[iu];
                g->intercept_edge_vx[off] = g->adj_edge_idx[iv];
                off++; iu++; iv++;
            } else if (x < y) {
                iu++;
            } else {
                iv++;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Public API: initialization                                         */
/* ------------------------------------------------------------------ */

// Construct a dress graph from an input edge list.
//
// 1. Builds raw adjacency (flat CSR) from the input edges.
// 2. Transforms it into the variant-specific adjacency (flat CSR).
// 3. Initializes all dress values to 2.0 (fixed-point identity).
// 4. Optionally precomputes intercepts for faster iteration.
//
// Takes ownership of U and V (freed by free_dress_graph).
// W is read but not retained; pass NULL for unweighted graphs.
p_dress_graph_t init_dress_graph(int N, int E, int *U, int *V,
                                 double *W, dress_variant_t variant,
                                 int precompute_intercepts)
{
    p_dress_graph_t g = (p_dress_graph_t)malloc(sizeof(dress_graph_t));
    int e;

    g->N = N;
    g->E = E;
    g->U = U;
    g->V = V;
    g->variant = variant;
    g->precompute_intercepts = precompute_intercepts;
    g->node_dress = (double *)malloc(N * sizeof(double));

    // Build raw then variant adjacency (both use flat CSR internally)
    int *raw_offset;
    adj_edge_t *raw_data;
    build_raw_adjacency(g, &raw_offset, &raw_data, W);
    build_variant_adjacency(g, raw_offset, raw_data);
    // raw_offset and raw_data are freed inside build_variant_adjacency

    // Initialize dress double-buffer to 2.0 (the fixed-point starting value)
    g->edge_dress      = (double *)malloc(E * sizeof(double));
    g->edge_dress_next = (double *)malloc(E * sizeof(double));
    for (e = 0; e < E; e++) {
        g->edge_dress[e]      = 1.0;
        g->edge_dress_next[e] = 1.0;
    }

    if (g->precompute_intercepts)
        compute_intercepts(g);

    return g;
}

/* ------------------------------------------------------------------ */
/*  dress computation                                                  */
/* ------------------------------------------------------------------ */

// Compute the dress value for a single edge e = (u,v).
//
// dress(u,v) = [ ∑_{x ∈ N[u]∩N[v]} (w_ux · d_ux + w_vx · d_vx)
//                + 2 · w_uv · d_uv + 8 ]
//              / (‖u‖ · ‖v‖)
//
// where ‖u‖ = sqrt( 4 + ∑_x w_ux · d_ux )  (self-loop contributes 4),
// the numerator constant 8 accounts for both self-loop × self-loop terms,
// and the +2·w_uv·d_uv term adds the u-v edge's own contribution from
// both sides (it also appears in the intercept but is handled separately
// since the intercept walk excludes it when u and v are not mutual neighbors
// via other paths).
// In the directed variants, only one of u or v has the other as a neighbor, so
// only one self-loop contributes and the constant is 4 instead of 8, and the
// cross-term is w_uv·d_uv instead of 2·w_uv·d_uv.
//
// Writes the result into edge_dress_next[e] (double-buffer).
static double fit_impl(p_dress_graph_t g, int e)
{
    int    u = g->U[e], v = g->V[e];
    double numerator   = 0;
    double denominator = g->node_dress[u] * g->node_dress[v];

    if (g->precompute_intercepts) {
        // O(|N[u] ∩ N[v]|) path: iterate only over precomputed common neighbors
        int off = g->intercept_offset[e];
        int end = g->intercept_offset[e + 1];
        for (int k = off; k < end; k++) {
            int eu = g->intercept_edge_ux[k];
            int ev = g->intercept_edge_vx[k];
                numerator += g->edge_weight[eu] * g->edge_dress[eu]
                           + g->edge_weight[ev] * g->edge_dress[ev];
        }
    } else {
        // O(deg_u + deg_v) path: sorted-merge walk to find common neighbors
        int iu = g->adj_offset[u], iu_end = g->adj_offset[u + 1];
        int iv = g->adj_offset[v], iv_end = g->adj_offset[v + 1];

        while (iu < iu_end && iv < iv_end) {
            int x = g->adj_target[iu], y = g->adj_target[iv];
            if (x == y) {
                int eu = g->adj_edge_idx[iu];
                int ev = g->adj_edge_idx[iv];
                    numerator += g->edge_weight[eu] * g->edge_dress[eu]
                               + g->edge_weight[ev] * g->edge_dress[ev];
                ++iu; ++iv;
            } else if (x < y) {
                ++iu;
            } else {
                ++iv;
            }
        }
    }

    // Add self-loop and edge cross-terms for nodes u and v.
    //
    // UNDIRECTED/DIRECTED: both u∈N[v] and v∈N[u] always hold, so both
    // self-loops cross:  (4 + w_vu·d) + (w_uv·d + 4) = 8 + 2·w·d.
    //
    // FORWARD/BACKWARD: only v∈N[u] (u→v exists); u∉N[v] in general
    // (v→u absent), so only v's self-loop crosses: (w_uv·d + 4) = 4 + w·d.
    double uv = g->edge_weight[e] * g->edge_dress[e];
    if (g->variant == DRESS_VARIANT_FORWARD || g->variant == DRESS_VARIANT_BACKWARD) {
        numerator += 4.0 + uv;
    } else {
        numerator += 8.0 + 2.0 * uv;
    }
    double dress_uv = denominator > 0.0
                    ? numerator / denominator
                    : 0.0;

    g->edge_dress_next[e] = dress_uv;
    return dress_uv;
}

/* ------------------------------------------------------------------ */
/*  Iterative fitting                                                  */
/* ------------------------------------------------------------------ */

// Run dress iterative fixed-point fitting.
//
// Each iteration:
//   1. Recompute node_dress[u] = sqrt(4 + ∑ w_ux · d_ux) for all u.
//   2. Recompute edge_dress_next[e] = fit_impl(e) for all e.
//   3. Swap edge_dress ↔ edge_dress_next (double-buffer).
//   4. Stop if max |d_old - d_new| < epsilon.
//
// On return:
//   *iterations = number of iterations performed (or max_iterations)
//   *delta      = final maximum per-edge change (only set on early stop)
void fit(p_dress_graph_t g, int max_iterations, double epsilon,
         int *iterations, double *delta)
{
    int iter;

    for (iter = 0; iter < max_iterations; ++iter) {
        double max_delta = 0.0;

        // Phase 1: compute per-node dress norm
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int u = 0; u < g->N; u++) {
            double dress_u = 4.0; // self-loop contribution
            int base = g->adj_offset[u];
            int end  = g->adj_offset[u + 1];
            for (int i = base; i < end; i++) {
                int ei = g->adj_edge_idx[i];
                dress_u += g->edge_weight[ei] * g->edge_dress[ei];
            }
            g->node_dress[u] = sqrt(dress_u);
        }

        // Phase 2: compute next dress value for every edge
#ifdef _OPENMP
        #pragma omp parallel for reduction(max:max_delta)
#endif
        for (int e = 0; e < g->E; e++) {
            double prev = g->edge_dress[e];
            double next = fit_impl(g, e);
            double d    = fabs(prev - next);
            if (d > max_delta)
                max_delta = d;
        }

        // Phase 3: swap double-buffer pointers
        double *tmp        = g->edge_dress;
        g->edge_dress      = g->edge_dress_next;
        g->edge_dress_next = tmp;

        // Phase 4: convergence check
        if (delta)      *delta      = max_delta;

        if (max_delta < epsilon) {
            if (iterations) *iterations = iter;
            return;
        }
    }

    if (iterations) *iterations = max_iterations;
}

/* ------------------------------------------------------------------ */
/*  Cleanup                                                            */
/* ------------------------------------------------------------------ */

// Free all heap memory owned by the dress graph, including U and V.
void free_dress_graph(p_dress_graph_t g)
{
    // CSR adjacency
    free(g->adj_offset);
    free(g->adj_target);
    free(g->adj_edge_idx);

    // Per-edge arrays
    free(g->edge_weight);
    free(g->edge_dress);
    free(g->edge_dress_next);

    // Per-node arrays
    free(g->node_dress);

    // Precomputed intercepts (conditional)
    if (g->precompute_intercepts) {
        free(g->intercept_offset);
        free(g->intercept_edge_ux);
        free(g->intercept_edge_vx);
    }

    // Input edge list (ownership transferred at init)
    free(g->U);
    free(g->V);
    free(g);
}