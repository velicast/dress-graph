/**
 * dress_mpi.h — MPI-distributed Δ^k-DRESS for libdress.
 *
 * Distributes the C(N,k) subgraph enumeration across MPI ranks using
 * the strided API: each rank processes subgraphs where
 * index % nprocs == rank, then the per-rank histograms are summed via
 * MPI_Allreduce.  All MPI logic lives in C — language bindings are
 * thin FFI calls.
 *
 * CPU usage:
 *
 *   #include "dress/dress.h"
 *   #include "dress/mpi/dress_mpi.h"
 *
 *   MPI_Init(&argc, &argv);
 *   p_dress_graph_t g = init_dress_graph(N, E, U, V, NULL, 0, 1);
 *   int64_t *hist = delta_dress_fit_mpi(g, 2, 100, 1e-6,
 *                                       &hist_size, 0, NULL,
 *                                       &num_sub, MPI_COMM_WORLD);
 *   free(hist);
 *   free_dress_graph(g);
 *   MPI_Finalize();
 *
 * GPU + MPI:
 *
 *   #include "dress/cuda/dress_cuda.h"
 *   #include "dress/mpi/dress_mpi.h"
 *
 *   delta_dress_fit_mpi_cuda(g, 2, 100, 1e-6,
 *                            &hist_size, 0, NULL,
 *                            &num_sub, MPI_COMM_WORLD);
 *
 * FFI callers (Python, Rust, Julia, R) use the _fcomm variants which
 * accept a Fortran MPI communicator handle (int) instead of MPI_Comm:
 *
 *   int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);
 *   delta_dress_fit_mpi_fcomm(g, ..., comm_f);
 */

#ifndef DRESS_MPI_H
#define DRESS_MPI_H

#include "dress/dress.h"
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── CPU + MPI ──────────────────────────────────────────────────── */

/**
 * MPI-distributed Δ^k-DRESS (CPU backend).
 *
 * Same parameters as delta_dress_fit plus an MPI communicator.
 * Each rank computes a local histogram for its stride of subgraphs.
 * The returned histogram is the global sum (identical on all ranks).
 * Caller must free() the returned pointer.
 *
 * When keep_multisets is non-zero, *multisets is set to a heap-allocated
 * C(N,k)*E matrix (same layout as delta_dress_fit).  Each rank computes
 * its own rows, then MPI_Allreduce(SUM) merges them (non-owned rows are
 * zero-initialised).  Caller must free(*multisets).
 */
int64_t *delta_dress_fit_mpi(
    p_dress_graph_t g, int k, int iterations,
    double epsilon, int *hist_size,
    int keep_multisets, double **multisets,
    int64_t *num_subgraphs,
    MPI_Comm comm);

/**
 * FFI-friendly variant: takes a Fortran MPI communicator handle (int)
 * instead of MPI_Comm.  Converts via MPI_Comm_f2c() internally.
 *
 * Python:  comm.py2f()          → comm_f
 * Julia:   MPI.Comm_c2f(comm)   → comm_f
 * Rust:    MPI_Comm_c2f(raw)    → comm_f
 * R:       pbdMPI::comm.c2f(0L) → comm_f
 */
int64_t *delta_dress_fit_mpi_fcomm(
    p_dress_graph_t g, int k, int iterations,
    double epsilon, int *hist_size,
    int keep_multisets, double **multisets,
    int64_t *num_subgraphs,
    int comm_f);

/* ── CUDA + MPI ─────────────────────────────────────────────────── */

#if defined(DRESS_CUDA) || defined(DRESS_CUDA_H)

/**
 * MPI-distributed Δ^k-DRESS (GPU backend).
 *
 * Same interface as delta_dress_fit_mpi but each rank fits its
 * subgraphs on the GPU via dress_fit_cuda.
 */
int64_t *delta_dress_fit_mpi_cuda(
    p_dress_graph_t g, int k, int iterations,
    double epsilon, int *hist_size,
    int keep_multisets, double **multisets,
    int64_t *num_subgraphs,
    MPI_Comm comm);

/** FFI-friendly CUDA + MPI variant — Fortran comm handle. */
int64_t *delta_dress_fit_mpi_cuda_fcomm(
    p_dress_graph_t g, int k, int iterations,
    double epsilon, int *hist_size,
    int keep_multisets, double **multisets,
    int64_t *num_subgraphs,
    int comm_f);

#endif /* DRESS_CUDA || DRESS_CUDA_H */

/* ── FFI helpers — no MPI package required in wrappers ──────────── */

/**
 * Initialise MPI (safe to call multiple times, calls MPI_Init if needed).
 * Must be invoked before any delta_dress_fit_mpi_* call.
 */
void dress_mpi_init(void);

/** Finalise MPI (safe to call multiple times). */
void dress_mpi_finalize(void);

/** Return rank of current process in MPI_COMM_WORLD. */
int dress_mpi_rank(void);

/** Return total number of MPI processes in MPI_COMM_WORLD. */
int dress_mpi_size(void);

/**
 * COMM_WORLD convenience — CPU backend.
 * Same as delta_dress_fit_mpi(..., MPI_COMM_WORLD).
 */
int64_t *delta_dress_fit_mpi_world(
    p_dress_graph_t g, int k, int iterations,
    double epsilon, int *hist_size,
    int keep_multisets, double **multisets,
    int64_t *num_subgraphs);

#if defined(DRESS_CUDA) || defined(DRESS_CUDA_H)
/**
 * COMM_WORLD convenience — CUDA backend.
 * Same as delta_dress_fit_mpi_cuda(..., MPI_COMM_WORLD).
 */
int64_t *delta_dress_fit_mpi_cuda_world(
    p_dress_graph_t g, int k, int iterations,
    double epsilon, int *hist_size,
    int keep_multisets, double **multisets,
    int64_t *num_subgraphs);
#endif /* DRESS_CUDA || DRESS_CUDA_H */

#ifdef __cplusplus
}
#endif

#endif /* DRESS_MPI_H */
