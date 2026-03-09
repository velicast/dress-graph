/*
 * dress_igraph_mpi.c — MPI-distributed Δ^k-DRESS igraph wrapper.
 *
 * Bridges igraph_t graphs to the MPI-distributed DRESS C API by
 * extracting the edge list / weights and delegating to
 * delta_dress_fit_mpi (CPU) or delta_dress_fit_mpi_cuda (GPU).
 *
 * Build (from repo root):
 *   mpicc -O3 -c dress_igraph_mpi.c -o dress_igraph_mpi.o \
 *       $(pkg-config --cflags igraph) \
 *       -I ../libdress/include -I ../libdress-igraph/include
 */

#include <dress/igraph/dress.h>
#include <dress/dress.h>
#include "dress/mpi/dress_mpi.h"

#include <igraph/igraph.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── internal: shared logic for CPU and CUDA MPI variants ───────── */

static int _mpi_igraph_impl(const igraph_t *graph,
                            const char *weight_attr,
                            dress_variant_t variant,
                            int k, int max_iters, double epsilon,
                            int precompute,
                            delta_dress_result_igraph_t *result,
                            MPI_Comm comm,
                            int use_cuda)
{
    if (!graph || !result) return -1;

    memset(result, 0, sizeof(*result));

    int N = (int)igraph_vcount(graph);
    int E = (int)igraph_ecount(graph);

    if (E == 0) {
        int nbins = (int)(2.0 / epsilon) + 1;
        result->hist_size = nbins;
        result->histogram = (int64_t *)calloc((size_t)nbins, sizeof(int64_t));
        return result->histogram ? 0 : -1;
    }

    /* ----- Allocate edge arrays (ownership transferred to DRESS) ----- */

    int    *U = (int *)malloc(E * sizeof(int));
    int    *V = (int *)malloc(E * sizeof(int));
    double *W = NULL;

    if (!U || !V) { free(U); free(V); return -1; }

    for (int e = 0; e < E; e++) {
        U[e] = (int)IGRAPH_FROM(graph, e);
        V[e] = (int)IGRAPH_TO(graph, e);
    }

    if (weight_attr != NULL &&
        igraph_cattribute_has_attr(graph, IGRAPH_ATTRIBUTE_EDGE, weight_attr))
    {
        W = (double *)malloc(E * sizeof(double));
        if (!W) { free(U); free(V); return -1; }
        for (int e = 0; e < E; e++)
            W[e] = igraph_cattribute_EAN(graph, weight_attr, e);
    }

    /* init_dress_graph takes ownership of U, V, W */
    p_dress_graph_t dg = init_dress_graph(N, E, U, V, W, variant, precompute);
    if (!dg) return -1;

    int hist_size = 0;
    int64_t *histogram = NULL;

#ifdef DRESS_CUDA
    if (use_cuda) {
        histogram = delta_dress_fit_mpi_cuda(
            dg, k, max_iters, epsilon, &hist_size,
            0, NULL, NULL, comm);
    } else
#endif
    {
        (void)use_cuda;
        histogram = delta_dress_fit_mpi(
            dg, k, max_iters, epsilon, &hist_size,
            0, NULL, NULL, comm);
    }

    free_dress_graph(dg);

    if (!histogram) return -1;

    result->histogram = histogram;
    result->hist_size = hist_size;
    return 0;
}

/* ── CPU + MPI ──────────────────────────────────────────────────── */

int delta_dress_fit_mpi_igraph(const igraph_t *graph,
                               const char *weight_attr,
                               dress_variant_t variant,
                               int k, int max_iters, double epsilon,
                               int precompute,
                               delta_dress_result_igraph_t *result,
                               MPI_Comm comm)
{
    return _mpi_igraph_impl(graph, weight_attr, variant, k, max_iters,
                            epsilon, precompute, result, comm, 0);
}

int delta_dress_fit_mpi_igraph_fcomm(const igraph_t *graph,
                                     const char *weight_attr,
                                     dress_variant_t variant,
                                     int k, int max_iters, double epsilon,
                                     int precompute,
                                     delta_dress_result_igraph_t *result,
                                     int comm_f)
{
    return delta_dress_fit_mpi_igraph(graph, weight_attr, variant, k,
                                     max_iters, epsilon, precompute, result,
                                     MPI_Comm_f2c(comm_f));
}

/* ── CUDA + MPI ─────────────────────────────────────────────────── */

#ifdef DRESS_CUDA

int delta_dress_fit_mpi_cuda_igraph(const igraph_t *graph,
                                    const char *weight_attr,
                                    dress_variant_t variant,
                                    int k, int max_iters, double epsilon,
                                    int precompute,
                                    delta_dress_result_igraph_t *result,
                                    MPI_Comm comm)
{
    return _mpi_igraph_impl(graph, weight_attr, variant, k, max_iters,
                            epsilon, precompute, result, comm, 1);
}

int delta_dress_fit_mpi_cuda_igraph_fcomm(const igraph_t *graph,
                                          const char *weight_attr,
                                          dress_variant_t variant,
                                          int k, int max_iters, double epsilon,
                                          int precompute,
                                          delta_dress_result_igraph_t *result,
                                          int comm_f)
{
    return delta_dress_fit_mpi_cuda_igraph(graph, weight_attr, variant, k,
                                          max_iters, epsilon, precompute,
                                          result, MPI_Comm_f2c(comm_f));
}

#endif /* DRESS_CUDA */
