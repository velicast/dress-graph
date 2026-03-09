/*
 * dress_mpi.c — MPI-distributed Δ^k-DRESS implementation.
 *
 * Provides linkable (non-inline) versions of the MPI distribution
 * functions so every language binding can call them through FFI.
 *
 * Build: compile with -DDRESS_CUDA to include the GPU+MPI variants.
 *        Link against libmpi (or equivalent).
 */

#include "dress/dress.h"
#include <mpi.h>
#include <stdlib.h>

#ifdef DRESS_CUDA
#include "dress/cuda/dress_cuda.h"
#endif

/* ── internal: strided fit + Allreduce ──────────────────────────── */

static int64_t *_mpi_reduce(
    p_dress_graph_t g, int k, int iterations,
    double epsilon, int *hist_size,
    int keep_multisets, double **multisets,
    int64_t *num_subgraphs,
    MPI_Comm comm,
    int use_cuda)
{
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    double *local_ms = NULL;
    int local_nbins = 0;
    int64_t *local_hist;

#ifdef DRESS_CUDA
    if (use_cuda) {
        local_hist = delta_dress_fit_cuda_strided(
            g, k, iterations, epsilon, &local_nbins,
            keep_multisets, keep_multisets ? &local_ms : NULL,
            num_subgraphs, rank, nprocs);
    } else
#endif
    {
        (void)use_cuda;
        local_hist = delta_dress_fit_strided(
            g, k, iterations, epsilon, &local_nbins,
            keep_multisets, keep_multisets ? &local_ms : NULL,
            num_subgraphs, rank, nprocs);
    }

    if (!local_hist) return NULL;

    int64_t *global_hist = (int64_t *)calloc(local_nbins, sizeof(int64_t));
    MPI_Allreduce(local_hist, global_hist, local_nbins,
                  MPI_INT64_T, MPI_SUM, comm);

    if (keep_multisets && multisets && local_ms && num_subgraphs) {
        int E = g->E;
        int64_t cnk = *num_subgraphs;
        double *global_ms = (double *)calloc((size_t)cnk * E, sizeof(double));
        MPI_Allreduce(local_ms, global_ms, (int)(cnk * E),
                      MPI_DOUBLE, MPI_SUM, comm);
        free(local_ms);
        *multisets = global_ms;
    } else if (multisets) {
        *multisets = NULL;
    }

    if (hist_size) *hist_size = local_nbins;

    free(local_hist);
    return global_hist;
}

/* ── CPU + MPI ──────────────────────────────────────────────────── */

int64_t *delta_dress_fit_mpi(
    p_dress_graph_t g, int k, int iterations,
    double epsilon, int *hist_size,
    int keep_multisets, double **multisets,
    int64_t *num_subgraphs,
    MPI_Comm comm)
{
    return _mpi_reduce(g, k, iterations, epsilon, hist_size,
                       keep_multisets, multisets, num_subgraphs, comm, 0);
}

int64_t *delta_dress_fit_mpi_fcomm(
    p_dress_graph_t g, int k, int iterations,
    double epsilon, int *hist_size,
    int keep_multisets, double **multisets,
    int64_t *num_subgraphs,
    int comm_f)
{
    return delta_dress_fit_mpi(g, k, iterations, epsilon, hist_size,
                               keep_multisets, multisets, num_subgraphs,
                               MPI_Comm_f2c(comm_f));
}

/* ── CUDA + MPI ─────────────────────────────────────────────────── */

#ifdef DRESS_CUDA

int64_t *delta_dress_fit_mpi_cuda(
    p_dress_graph_t g, int k, int iterations,
    double epsilon, int *hist_size,
    int keep_multisets, double **multisets,
    int64_t *num_subgraphs,
    MPI_Comm comm)
{
    return _mpi_reduce(g, k, iterations, epsilon, hist_size,
                       keep_multisets, multisets, num_subgraphs, comm, 1);
}

int64_t *delta_dress_fit_mpi_cuda_fcomm(
    p_dress_graph_t g, int k, int iterations,
    double epsilon, int *hist_size,
    int keep_multisets, double **multisets,
    int64_t *num_subgraphs,
    int comm_f)
{
    return delta_dress_fit_mpi_cuda(g, k, iterations, epsilon, hist_size,
                                    keep_multisets, multisets, num_subgraphs,
                                    MPI_Comm_f2c(comm_f));
}

#endif /* DRESS_CUDA */

/* ── FFI helpers — no MPI package needed in wrappers ────────────── */

void dress_mpi_init(void)
{
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) MPI_Init(NULL, NULL);
}

void dress_mpi_finalize(void)
{
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Finalize();
}

int dress_mpi_rank(void)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int dress_mpi_size(void)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

/* COMM_WORLD convenience — CPU */
int64_t *delta_dress_fit_mpi_world(
    p_dress_graph_t g, int k, int iterations,
    double epsilon, int *hist_size,
    int keep_multisets, double **multisets,
    int64_t *num_subgraphs)
{
    return delta_dress_fit_mpi(g, k, iterations, epsilon, hist_size,
                               keep_multisets, multisets, num_subgraphs,
                               MPI_COMM_WORLD);
}

#ifdef DRESS_CUDA
/* COMM_WORLD convenience — CUDA */
int64_t *delta_dress_fit_mpi_cuda_world(
    p_dress_graph_t g, int k, int iterations,
    double epsilon, int *hist_size,
    int keep_multisets, double **multisets,
    int64_t *num_subgraphs)
{
    return delta_dress_fit_mpi_cuda(g, k, iterations, epsilon, hist_size,
                                    keep_multisets, multisets, num_subgraphs,
                                    MPI_COMM_WORLD);
}
#endif /* DRESS_CUDA */
