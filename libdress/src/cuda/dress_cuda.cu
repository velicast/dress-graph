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
/*  Kernel 1: node_dress                                               */
/*                                                                     */
/*  node_dress[u] = sqrt( 4 + Σ edge_weight[ei] * edge_dress[ei] )    */
/*  where the sum runs over all half-edges in u's CSR row.             */
/* ------------------------------------------------------------------ */

__global__ void
kernel_node_dress(const int    N,
                  const int   * __restrict__ adj_offset,
                  const int   * __restrict__ adj_edge_idx,
                  const double* __restrict__ edge_weight,
                  const double* __restrict__ edge_dress,
                  double      * __restrict__ node_dress)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N) return;

    double sum = 4.0;                       /* self-loop contribution */
    int base = adj_offset[u];
    int end  = adj_offset[u + 1];
    for (int i = base; i < end; i++) {
        int ei = adj_edge_idx[i];
        sum += edge_weight[ei] * edge_dress[ei];
    }
    node_dress[u] = sqrt(sum);
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
        const int   * __restrict__ U,
        const int   * __restrict__ V,
        const double* __restrict__ edge_weight,
        const double* __restrict__ edge_dress,
        const double* __restrict__ node_dress,
        const int   * __restrict__ intercept_offset,
        const int   * __restrict__ intercept_edge_ux,
        const int   * __restrict__ intercept_edge_vx,
        double      * __restrict__ edge_dress_next,
        unsigned long long * __restrict__ d_max_delta)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int u = U[e], v = V[e];
    double numerator = 0.0;

    /* Intercept walk — O(|N[u] ∩ N[v]|) */
    int off = intercept_offset[e];
    int end = intercept_offset[e + 1];
    for (int k = off; k < end; k++) {
        int eu = intercept_edge_ux[k];
        int ev = intercept_edge_vx[k];
        numerator += edge_weight[eu] * edge_dress[eu]
                   + edge_weight[ev] * edge_dress[ev];
    }

    /* Self-loop + edge cross-terms */
    double uv = edge_weight[e] * edge_dress[e];
    if (variant == 2 /* FORWARD */ || variant == 3 /* BACKWARD */) {
        numerator += 4.0 + uv;
    } else {
        numerator += 8.0 + 2.0 * uv;
    }

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
        const int   * __restrict__ U,
        const int   * __restrict__ V,
        const double* __restrict__ edge_weight,
        const double* __restrict__ edge_dress,
        const double* __restrict__ node_dress,
        const int   * __restrict__ adj_offset,
        const int   * __restrict__ adj_target,
        const int   * __restrict__ adj_edge_idx,
        double      * __restrict__ edge_dress_next,
        unsigned long long * __restrict__ d_max_delta)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int u = U[e], v = V[e];
    double numerator = 0.0;

    /* Sorted-merge walk — O(deg_u + deg_v) */
    int iu     = adj_offset[u], iu_end = adj_offset[u + 1];
    int iv     = adj_offset[v], iv_end = adj_offset[v + 1];

    while (iu < iu_end && iv < iv_end) {
        int x = adj_target[iu], y = adj_target[iv];
        if (x == y) {
            int eu = adj_edge_idx[iu];
            int ev = adj_edge_idx[iv];
            numerator += edge_weight[eu] * edge_dress[eu]
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
        numerator += 4.0 + uv;
    } else {
        numerator += 8.0 + 2.0 * uv;
    }

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

    /* ── Kernel launch config ──────────────────────────────────── */

    const int BLOCK = 256;
    const int grid_N = (N + BLOCK - 1) / BLOCK;
    const int grid_E = (E + BLOCK - 1) / BLOCK;

    /* ── Iteration loop ────────────────────────────────────────── */

    for (int iter = 0; iter < max_iterations; iter++) {

        /* Phase 1: compute node_dress */
        kernel_node_dress<<<grid_N, BLOCK>>>(
            N,
            d_adj_offset, d_adj_edge_idx,
            d_edge_weight, d_edge_dress,
            d_node_dress);

        /* Reset max_delta to 0 */
        unsigned long long zero = 0ULL;
        CUDA_CHECK(cudaMemcpy(d_max_delta, &zero,
                              sizeof(zero), cudaMemcpyHostToDevice));

        /* Phase 2: compute edge_dress_next */
        if (use_intercepts) {
            kernel_edge_dress_intercept<<<grid_E, BLOCK>>>(
                E, variant,
                d_U, d_V,
                d_edge_weight, d_edge_dress, d_node_dress,
                d_intercept_offset,
                d_intercept_edge_ux, d_intercept_edge_vx,
                d_edge_dress_next,
                d_max_delta);
        } else {
            kernel_edge_dress_merge<<<grid_E, BLOCK>>>(
                E, variant,
                d_U, d_V,
                d_edge_weight, d_edge_dress, d_node_dress,
                d_adj_offset, d_adj_target, d_adj_edge_idx,
                d_edge_dress_next,
                d_max_delta);
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
}
