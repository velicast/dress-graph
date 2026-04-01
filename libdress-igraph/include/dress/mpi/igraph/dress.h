/**
 * dress/mpi/igraph/dress.h — MPI-distributed DRESS igraph wrapper via
 *                             include-based switching.
 *
 * Drop-in replacement for dress/igraph/dress.h.
 * Including this header redirects dress_delta_fit() to the
 * MPI-distributed implementation — no source changes required:
 *
 *   // Single-process (CPU)
 *   #include <dress/igraph/dress.h>
 *   dress_delta_fit(&graph, NULL, DRESS_VARIANT_UNDIRECTED,
 *                   1, 100, 1e-6, 1, &result);
 *
 *   // MPI-distributed (CPU) — same call, different include
 *   #include <dress/mpi/igraph/dress.h>
 *   dress_delta_fit(&graph, NULL, DRESS_VARIANT_UNDIRECTED,
 *                   1, 100, 1e-6, 1, &result);
 *
 *   // MPI-distributed (GPU) — include CUDA header first
 *   #include <dress/cuda/igraph/dress.h>
 *   #include <dress/mpi/igraph/dress.h>
 *   dress_delta_fit(&graph, NULL, DRESS_VARIANT_UNDIRECTED,
 *                   1, 100, 1e-6, 1, &result);
 *
 * The redirect macros append MPI_COMM_WORLD as the communicator.
 * For a custom communicator, call dress_delta_fit_mpi_igraph() or
 * dress_delta_fit_mpi_cuda_igraph() directly.
 *
 * Do not include both this header and dress/igraph/dress.h in the same
 * translation unit — the macros will conflict.
 */
#ifndef DRESS_IGRAPH_MPI_REDIRECT_H
#define DRESS_IGRAPH_MPI_REDIRECT_H

#include <dress/igraph/dress.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── CPU + MPI igraph function declarations ────────────────────── */

/**
 * MPI-distributed Δ^k-DRESS on an igraph graph (CPU backend).
 *
 * Same parameters as dress_delta_fit_igraph plus an MPI communicator.
 * Uses the same result type (delta_dress_result_igraph_t); free with
 * delta_dress_free_igraph().
 */
int dress_delta_fit_mpi_igraph(const igraph_t *graph,
                               const char *weight_attr,
                               const char *vertex_weight_attr,
                               dress_variant_t variant,
                               int k, int max_iters,
                               double epsilon,
                               int n_samples,
                               unsigned int seed,
                               int precompute,
                               int keep_multisets,
                               int compute_histogram,
                               delta_dress_result_igraph_t *result,
                               MPI_Comm comm);

int dress_delta_fit_mpi_igraph_fcomm(const igraph_t *graph,
                                     const char *weight_attr,
                                     const char *vertex_weight_attr,
                                     dress_variant_t variant,
                                     int k, int max_iters,
                                     double epsilon,
                                     int n_samples,
                                     unsigned int seed,
                                     int precompute,
                                     int keep_multisets,
                                     int compute_histogram,
                                     delta_dress_result_igraph_t *result,
                                     int comm_f);

/* ── CUDA + MPI igraph function declarations ───────────────────── */

#if defined(DRESS_CUDA) || defined(DRESS_CUDA_H)

int dress_delta_fit_mpi_cuda_igraph(const igraph_t *graph,
                                    const char *weight_attr,
                                    const char *vertex_weight_attr,
                                    dress_variant_t variant,
                                    int k, int max_iters,
                                    double epsilon,
                               int n_samples,
                               unsigned int seed,
                               int precompute,
                               int keep_multisets,
                               int compute_histogram,
                               delta_dress_result_igraph_t *result,
                               MPI_Comm comm);

int dress_delta_fit_mpi_cuda_igraph_fcomm(const igraph_t *graph,
                                          const char *weight_attr,
                                          const char *vertex_weight_attr,
                                          dress_variant_t variant,
                                          int k, int max_iters,
                                          double epsilon,
                                     int n_samples,
                                     unsigned int seed,
                                     int precompute,
                                     int keep_multisets,
                                     int compute_histogram,
                                     delta_dress_result_igraph_t *result,
                                     int comm_f);

#endif /* DRESS_CUDA || DRESS_CUDA_H */

#ifdef __cplusplus
}
#endif

/* ── Redirect dress_delta_fit to MPI implementation ─────────────── */

#undef dress_delta_fit

#ifdef DRESS_IGRAPH_CUDA_REDIRECT_H

#define dress_delta_fit(g, w, nw, v, k, mi, eps, nsamp, sd, pc, km, ch, res) \
    dress_delta_fit_mpi_cuda_igraph((g), (w), (nw), (v), (k), (mi), (eps), \
                                    (nsamp), (sd), (pc), (km), (ch), (res), MPI_COMM_WORLD)

#else

#define dress_delta_fit(g, w, nw, v, k, mi, eps, nsamp, sd, pc, km, ch, res) \
    dress_delta_fit_mpi_igraph((g), (w), (nw), (v), (k), (mi), (eps), \
                               (nsamp), (sd), (pc), (km), (ch), (res), MPI_COMM_WORLD)

#endif /* DRESS_IGRAPH_CUDA_REDIRECT_H */

#endif /* DRESS_IGRAPH_MPI_REDIRECT_H */
