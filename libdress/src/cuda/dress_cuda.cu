/**
 * dress_cuda.cu — CUDA implementation of dress_fit() for libdress.
 *
 * Exact algorithmic match with the CPU fit() in dress.c:
 *
 *   Per iteration:
 *     Phase 1 — node_dress[u] = sqrt(4 + Σ edge_weight[e]*edge_dress[e])
 *               over all half-edges in u's CSR segment.
 *               → One thread per node.
 *
 *     Phase 2 — For each edge e = (u,v), compute dress(u,v):
 *               ● With intercepts: walk intercept_edge_ux/vx arrays.
 *               ● Without intercepts: sorted-merge on adj_target.
 *               Add variant-specific self-loop/cross-term:
 *                 FORWARD|BACKWARD: 4 + w·d
 *                 UNDIRECTED|DIRECTED: 8 + 2·w·d
 *               Divide by node_dress[u]*node_dress[v].
 *               Write to edge_dress_next[e].
 *               Track max |d_old − d_new| via atomicMax.
 *               → One thread per edge.
 *
 *     Phase 3 — Swap edge_dress ↔ edge_dress_next (pointer swap on host).
 *     Phase 4 — Check convergence (download scalar max_delta from GPU).
 *
 * After the loop, download edge_dress[] and node_dress[] to the host graph.
 */

#include "dress/cuda/dress_cuda.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/*  Error checking                                                     */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t _err = (call);                                       \
        if (_err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Double-precision atomicMax via CAS                                 */
/* ------------------------------------------------------------------ */

static __device__ __forceinline__
void atomicMaxDouble(unsigned long long *addr, double val)
{
    unsigned long long val_bits = __double_as_longlong(val);
    unsigned long long old = *addr;
    unsigned long long assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val) return;
        old = atomicCAS(addr, assumed, val_bits);
    } while (assumed != old);
}

/* ------------------------------------------------------------------ */
/*  Label-independent summation: sort + KBN on GPU                     */
/* ------------------------------------------------------------------ */

// Insertion sort for small partitions (n ≤ 16).
static __device__ __forceinline__
void insertion_sort_d(double *arr, int lo, int hi)
{
    for (int i = lo + 1; i <= hi; i++) {
        double key = arr[i];
        int j = i - 1;
        while (j >= lo && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Median-of-three pivot selection.
static __device__ __forceinline__
int median3(double *arr, int lo, int hi)
{
    int mid = lo + (hi - lo) / 2;
    if (arr[lo] > arr[mid]) { double t = arr[lo]; arr[lo] = arr[mid]; arr[mid] = t; }
    if (arr[lo] > arr[hi])  { double t = arr[lo]; arr[lo] = arr[hi];  arr[hi]  = t; }
    if (arr[mid] > arr[hi]) { double t = arr[mid]; arr[mid] = arr[hi]; arr[hi] = t; }
    return mid;
}

// Iterative quicksort with median-of-three pivot and insertion sort cutoff.
// Stack depth ≤ 2·log₂(n) — 64 entries handles n up to 2^32.
static __device__ __forceinline__
void sort_d(double *arr, int n)
{
    if (n <= 16) { insertion_sort_d(arr, 0, n - 1); return; }

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;
    stack[sp++] = n - 1;

    while (sp > 0) {
        int hi = stack[--sp];
        int lo = stack[--sp];

        if (hi - lo < 16) {
            insertion_sort_d(arr, lo, hi);
            continue;
        }

        // Median-of-three pivot, place at hi-1
        int m = median3(arr, lo, hi);
        double t = arr[m]; arr[m] = arr[hi - 1]; arr[hi - 1] = t;
        double pivot = arr[hi - 1];

        // Hoare-style partition
        int i = lo, j = hi - 1;
        for (;;) {
            while (arr[++i] < pivot) {}
            while (arr[--j] > pivot) {}
            if (i >= j) break;
            t = arr[i]; arr[i] = arr[j]; arr[j] = t;
        }
        // Restore pivot
        arr[hi - 1] = arr[i]; arr[i] = pivot;

        // Push larger partition first (limits stack depth to log₂(n))
        int left_size  = i - lo;
        int right_size = hi - i;
        if (left_size > right_size) {
            if (left_size > 1)  { stack[sp++] = lo;    stack[sp++] = i - 1; }
            if (right_size > 1) { stack[sp++] = i + 1; stack[sp++] = hi;    }
        } else {
            if (right_size > 1) { stack[sp++] = i + 1; stack[sp++] = hi;    }
            if (left_size > 1)  { stack[sp++] = lo;    stack[sp++] = i - 1; }
        }
    }
}

// KBN compensated sum of a pre-sorted array.
static __device__ __forceinline__
double kbn_sum_d(const double *arr, int n)
{
    if (n == 0) return 0.0;
    double sum  = arr[0];
    double comp = 0.0;
    for (int i = 1; i < n; i++) {
        double t = sum + arr[i];
        if (fabs(sum) >= fabs(arr[i]))
            comp += (sum - t) + arr[i];
        else
            comp += (arr[i] - t) + sum;
        sum = t;
    }
    return sum + comp;
}

/* ------------------------------------------------------------------ */
/*  Kernel 1: node_dress                                               */
/*                                                                     */
/*  node_dress[u] = sqrt( 4 + Σ edge_weight[ei] * edge_dress[ei] )    */
/*  where the sum runs over all half-edges in u's CSR row.             */
/*                                                                     */
/*  Each thread gets a slice of d_work sized (max_degree+1) doubles    */
/*  for exact sort+KBN — no fixed buffer limit.                        */
/* ------------------------------------------------------------------ */

__global__ void
kernel_node_dress(const int    N,
                  const int    max_buf,
                  const int   * __restrict__ adj_offset,
                  const int   * __restrict__ adj_edge_idx,
                  const double* __restrict__ edge_weight,
                  const double* __restrict__ edge_dress,
                  double      * __restrict__ node_dress,
                  double      * __restrict__ d_work)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N) return;

    int base = adj_offset[u];
    int end  = adj_offset[u + 1];
    int deg  = end - base;

    double *buf = d_work + (size_t)u * max_buf;

    buf[0] = 4.0;  /* self-loop contribution */
    for (int i = 0; i < deg; i++) {
        int ei = adj_edge_idx[base + i];
        buf[i + 1] = edge_weight[ei] * edge_dress[ei];
    }

    int n = deg + 1;
    sort_d(buf, n);
    node_dress[u] = sqrt(kbn_sum_d(buf, n));
}

/* ------------------------------------------------------------------ */
/*  Kernel 2a: edge_dress — WITH precomputed intercepts                */
/*                                                                     */
/*  For edge e = (u,v):                                                */
/*    num  = Σ_{k ∈ intercepts(e)} (w_ux·d_ux + w_vx·d_vx)            */
/*         + self_term(variant, w_e, d_e)                              */
/*    edge_dress_next[e] = num / (node_dress[u] * node_dress[v])       */
/* ------------------------------------------------------------------ */

__global__ void
kernel_edge_dress_intercept(
        const int     E,
        const int     variant,           /* dress_variant_t as int */
        const int     max_buf,
        const int   * __restrict__ U,
        const int   * __restrict__ V,
        const double* __restrict__ edge_weight,
        const double* __restrict__ edge_dress,
        const double* __restrict__ node_dress,
        const int   * __restrict__ intercept_offset,
        const int   * __restrict__ intercept_edge_ux,
        const int   * __restrict__ intercept_edge_vx,
        double      * __restrict__ edge_dress_next,
        unsigned long long * __restrict__ d_max_delta,
        double      * __restrict__ d_work)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int u = U[e], v = V[e];

    double *buf = d_work + (size_t)e * max_buf;

    /* Collect numerator terms into workspace for sort+KBN */
    int off = intercept_offset[e];
    int end = intercept_offset[e + 1];
    int n = 0;

    for (int k = off; k < end; k++) {
        int eu = intercept_edge_ux[k];
        int ev = intercept_edge_vx[k];
        buf[n++] = edge_weight[eu] * edge_dress[eu]
                 + edge_weight[ev] * edge_dress[ev];
    }

    /* Self-loop + edge cross-terms */
    double uv = edge_weight[e] * edge_dress[e];
    if (variant == 2 /* FORWARD */ || variant == 3 /* BACKWARD */) {
        buf[n++] = 4.0 + uv;
    } else {
        buf[n++] = 8.0 + 2.0 * uv;
    }

    sort_d(buf, n);
    double numerator = kbn_sum_d(buf, n);

    double denom = node_dress[u] * node_dress[v];
    double dress_uv = (denom > 0.0) ? (numerator / denom) : 0.0;
    edge_dress_next[e] = dress_uv;

    /* Convergence tracking */
    double diff = fabs(edge_dress[e] - dress_uv);
    atomicMaxDouble(d_max_delta, diff);
}

/* ------------------------------------------------------------------ */
/*  Kernel 2b: edge_dress — WITHOUT intercepts (sorted-merge walk)     */
/*                                                                     */
/*  Same formula, but common neighbors are found via a sorted-merge    */
/*  walk on the adj_target arrays — matching the CPU non-intercept     */
/*  code path in dress.c exactly.                                      */
/* ------------------------------------------------------------------ */

__global__ void
kernel_edge_dress_merge(
        const int     E,
        const int     variant,
        const int     max_buf,
        const int   * __restrict__ U,
        const int   * __restrict__ V,
        const double* __restrict__ edge_weight,
        const double* __restrict__ edge_dress,
        const double* __restrict__ node_dress,
        const int   * __restrict__ adj_offset,
        const int   * __restrict__ adj_target,
        const int   * __restrict__ adj_edge_idx,
        double      * __restrict__ edge_dress_next,
        unsigned long long * __restrict__ d_max_delta,
        double      * __restrict__ d_work)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int u = U[e], v = V[e];

    double *buf = d_work + (size_t)e * max_buf;
    int n = 0;

    /* Collect numerator terms for sort+KBN */
    int iu     = adj_offset[u], iu_end = adj_offset[u + 1];
    int iv     = adj_offset[v], iv_end = adj_offset[v + 1];

    /* Sorted-merge walk — O(deg_u + deg_v) */
    while (iu < iu_end && iv < iv_end) {
        int x = adj_target[iu], y = adj_target[iv];
        if (x == y) {
            int eu = adj_edge_idx[iu];
            int ev = adj_edge_idx[iv];
            buf[n++] = edge_weight[eu] * edge_dress[eu]
                     + edge_weight[ev] * edge_dress[ev];
            ++iu; ++iv;
        } else if (x < y) {
            ++iu;
        } else {
            ++iv;
        }
    }

    /* Self-loop + edge cross-terms */
    double uv = edge_weight[e] * edge_dress[e];
    if (variant == 2 /* FORWARD */ || variant == 3 /* BACKWARD */) {
        buf[n++] = 4.0 + uv;
    } else {
        buf[n++] = 8.0 + 2.0 * uv;
    }

    sort_d(buf, n);
    double numerator = kbn_sum_d(buf, n);

    double denom = node_dress[u] * node_dress[v];
    double dress_uv = (denom > 0.0) ? (numerator / denom) : 0.0;
    edge_dress_next[e] = dress_uv;

    /* Convergence tracking */
    double diff = fabs(edge_dress[e] - dress_uv);
    atomicMaxDouble(d_max_delta, diff);
}

/* ------------------------------------------------------------------ */
/*  dress_fit_cuda — host-side iteration loop                          */
/* ------------------------------------------------------------------ */

void dress_fit_cuda(p_dress_graph_t g, int max_iterations, double epsilon,
                    int *iterations, double *delta)
{
    const int N = g->N;
    const int E = g->E;
    const int S = g->adj_offset[N];       /* total half-edges */
    const int variant = (int)g->variant;
    const int use_intercepts = g->precompute_intercepts;
    int T = 0;
    if (use_intercepts)
        T = g->intercept_offset[E];       /* total intercept entries */

    /* ── Upload arrays to GPU ──────────────────────────────────── */

    /* CSR adjacency (needed for both kernel paths) */
    int *d_adj_offset, *d_adj_edge_idx;
    CUDA_CHECK(cudaMalloc(&d_adj_offset,   (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_adj_edge_idx, S * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_adj_offset,   g->adj_offset,
                          (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adj_edge_idx, g->adj_edge_idx,
                          S * sizeof(int), cudaMemcpyHostToDevice));

    /* adj_target — only needed when NOT using intercepts */
    int *d_adj_target = NULL;
    if (!use_intercepts) {
        CUDA_CHECK(cudaMalloc(&d_adj_target, S * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_adj_target, g->adj_target,
                              S * sizeof(int), cudaMemcpyHostToDevice));
    }

    /* Intercept arrays — only when using intercepts */
    int *d_intercept_offset = NULL;
    int *d_intercept_edge_ux = NULL;
    int *d_intercept_edge_vx = NULL;
    if (use_intercepts) {
        CUDA_CHECK(cudaMalloc(&d_intercept_offset, (E + 1) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_intercept_offset, g->intercept_offset,
                              (E + 1) * sizeof(int), cudaMemcpyHostToDevice));
        if (T > 0) {
            CUDA_CHECK(cudaMalloc(&d_intercept_edge_ux, T * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_intercept_edge_vx, T * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_intercept_edge_ux, g->intercept_edge_ux,
                                  T * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_intercept_edge_vx, g->intercept_edge_vx,
                                  T * sizeof(int), cudaMemcpyHostToDevice));
        }
    }

    /* Edge arrays */
    int    *d_U, *d_V;
    double *d_edge_weight, *d_edge_dress, *d_edge_dress_next;
    CUDA_CHECK(cudaMalloc(&d_U,               E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_V,               E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_weight,     E * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_edge_dress,      E * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_edge_dress_next, E * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_U,           g->U,           E * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V,           g->V,           E * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edge_weight, g->edge_weight, E * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edge_dress,  g->edge_dress,  E * sizeof(double), cudaMemcpyHostToDevice));

    /* Node array */
    double *d_node_dress;
    CUDA_CHECK(cudaMalloc(&d_node_dress, N * sizeof(double)));

    /* Convergence scalar */
    unsigned long long *d_max_delta;
    CUDA_CHECK(cudaMalloc(&d_max_delta, sizeof(unsigned long long)));

    /* ── Per-thread workspaces for sort+KBN ─────────────────────── */

    /* Phase 1 (node kernel): each of N threads needs (max_degree+1) doubles */
    const int max_deg = g->max_degree;
    const int node_buf = max_deg + 1;  /* +1 for self-loop constant */

    double *d_work_node;
    CUDA_CHECK(cudaMalloc(&d_work_node, (size_t)N * node_buf * sizeof(double)));

    /* Phase 2 (edge kernel): each of E threads needs at most
     *   min(deg_u, deg_v) + 1 ≤ max_degree + 1 doubles.
     * We over-allocate to max_degree + 1 per thread for simplicity. */
    const int edge_buf = max_deg + 1;

    double *d_work_edge;
    CUDA_CHECK(cudaMalloc(&d_work_edge, (size_t)E * edge_buf * sizeof(double)));

    /* ── Kernel launch config ──────────────────────────────────── */

    const int BLOCK = 256;
    const int grid_N = (N + BLOCK - 1) / BLOCK;
    const int grid_E = (E + BLOCK - 1) / BLOCK;

    /* ── Iteration loop ────────────────────────────────────────── */

    for (int iter = 0; iter < max_iterations; iter++) {

        /* Phase 1: compute node_dress */
        kernel_node_dress<<<grid_N, BLOCK>>>(
            N, node_buf,
            d_adj_offset, d_adj_edge_idx,
            d_edge_weight, d_edge_dress,
            d_node_dress, d_work_node);

        /* Reset max_delta to 0 */
        unsigned long long zero = 0ULL;
        CUDA_CHECK(cudaMemcpy(d_max_delta, &zero,
                              sizeof(zero), cudaMemcpyHostToDevice));

        /* Phase 2: compute edge_dress_next */
        if (use_intercepts) {
            kernel_edge_dress_intercept<<<grid_E, BLOCK>>>(
                E, variant, edge_buf,
                d_U, d_V,
                d_edge_weight, d_edge_dress, d_node_dress,
                d_intercept_offset,
                d_intercept_edge_ux, d_intercept_edge_vx,
                d_edge_dress_next,
                d_max_delta, d_work_edge);
        } else {
            kernel_edge_dress_merge<<<grid_E, BLOCK>>>(
                E, variant, edge_buf,
                d_U, d_V,
                d_edge_weight, d_edge_dress, d_node_dress,
                d_adj_offset, d_adj_target, d_adj_edge_idx,
                d_edge_dress_next,
                d_max_delta, d_work_edge);
        }

        /* Phase 3: swap double buffers */
        double *tmp        = d_edge_dress;
        d_edge_dress       = d_edge_dress_next;
        d_edge_dress_next  = tmp;

        /* Phase 4: convergence check */
        unsigned long long max_delta_bits;
        CUDA_CHECK(cudaMemcpy(&max_delta_bits, d_max_delta,
                              sizeof(max_delta_bits), cudaMemcpyDeviceToHost));
        double max_delta;
        memcpy(&max_delta, &max_delta_bits, sizeof(double));

        if (delta) *delta = max_delta;

        if (max_delta < epsilon) {
            if (iterations) *iterations = iter;
            goto cleanup;
        }
    }

    if (iterations) *iterations = max_iterations;

    /* ── Download results + free GPU memory ────────────────────── */

cleanup:
    /* Download edge_dress (from whichever buffer is "current" after swap) */
    CUDA_CHECK(cudaMemcpy(g->edge_dress, d_edge_dress,
                          E * sizeof(double), cudaMemcpyDeviceToHost));

    /* Download node_dress as-is from the last Phase 1 computation.
     * This matches CPU fit() behavior exactly: node_dress is computed
     * from the edge_dress values BEFORE the final Phase 2 + swap. */
    CUDA_CHECK(cudaMemcpy(g->node_dress, d_node_dress,
                          N * sizeof(double), cudaMemcpyDeviceToHost));

    /* Free device memory */
    cudaFree(d_adj_offset);
    cudaFree(d_adj_edge_idx);
    if (d_adj_target)          cudaFree(d_adj_target);
    if (d_intercept_offset)    cudaFree(d_intercept_offset);
    if (d_intercept_edge_ux)   cudaFree(d_intercept_edge_ux);
    if (d_intercept_edge_vx)   cudaFree(d_intercept_edge_vx);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_edge_weight);
    cudaFree(d_edge_dress);
    cudaFree(d_edge_dress_next);
    cudaFree(d_node_dress);
    cudaFree(d_max_delta);
    cudaFree(d_work_node);
    cudaFree(d_work_edge);
}
