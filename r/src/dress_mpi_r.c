/*
 * dress_mpi_r.c -- R bridge for the MPI-distributed DRESS library.
 *
 * Same API as dress_r.c (delta_dress_fit) but calls the C MPI backend.
 * Exposed via the `mpi` environment in R:
 *
 *   library(dress)
 *   mpi$delta_dress_fit(4, sources, targets, k = 2L)
 *
 * Only compiled when DRESS_MPI is defined (requires MPI library).
 */

#ifdef DRESS_MPI

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include "dress/dress.h"
#include "dress/mpi/dress_mpi.h"

/* ------------------------------------------------------------------ */
/*  delta_dress_fit_mpi                                                */
/* ------------------------------------------------------------------ */
SEXP C_delta_dress_fit_mpi(SEXP n_vertices_,
                           SEXP sources_,
                           SEXP targets_,
                           SEXP weights_,
                           SEXP k_,
                           SEXP variant_,
                           SEXP max_iterations_,
                           SEXP epsilon_,
                           SEXP precompute_,
                           SEXP keep_multisets_,
                           SEXP comm_f_) {

    int N  = INTEGER(n_vertices_)[0];
    int E  = LENGTH(sources_);
    int k              = INTEGER(k_)[0];
    int variant        = INTEGER(variant_)[0];
    int max_iterations = INTEGER(max_iterations_)[0];
    double epsilon     = REAL(epsilon_)[0];
    int precompute     = INTEGER(precompute_)[0];
    int keep_ms        = INTEGER(keep_multisets_)[0];
    int comm_f         = INTEGER(comm_f_)[0];

    /* Allocate copies (init_dress_graph takes ownership). */
    int *U = (int *)malloc(E * sizeof(int));
    int *V = (int *)malloc(E * sizeof(int));
    if (!U || !V) {
        free(U); free(V);
        error("delta_dress_fit_mpi: memory allocation failed");
    }
    memcpy(U, INTEGER(sources_), E * sizeof(int));
    memcpy(V, INTEGER(targets_), E * sizeof(int));

    double *W = NULL;
    if (!isNull(weights_)) {
        W = (double *)malloc(E * sizeof(double));
        if (!W) { free(U); free(V); error("delta_dress_fit_mpi: memory allocation failed"); }
        memcpy(W, REAL(weights_), E * sizeof(double));
    }

    p_dress_graph_t g = init_dress_graph(N, E, U, V, W,
                                         (dress_variant_t)variant, precompute);
    if (!g) {
        error("delta_dress_fit_mpi: init_dress_graph returned NULL");
    }

    int hist_size = 0;
    double *ms_ptr = NULL;
    int64_t num_sub = 0;
    int64_t *hist = delta_dress_fit_mpi_fcomm(g, k, max_iterations, epsilon,
                                              &hist_size,
                                              keep_ms,
                                              keep_ms ? &ms_ptr : NULL,
                                              keep_ms ? &num_sub : NULL,
                                              comm_f);

    /* Build result list: histogram + hist_size [+ multisets + num_subgraphs] */
    int n_fields = keep_ms ? 4 : 2;
    SEXP result = PROTECT(allocVector(VECSXP, n_fields));
    SEXP names  = PROTECT(allocVector(STRSXP, n_fields));

    SET_STRING_ELT(names, 0, mkChar("histogram"));
    SET_STRING_ELT(names, 1, mkChar("hist_size"));
    if (keep_ms) {
        SET_STRING_ELT(names, 2, mkChar("multisets"));
        SET_STRING_ELT(names, 3, mkChar("num_subgraphs"));
    }
    setAttrib(result, R_NamesSymbol, names);

    /* histogram — numeric vector (R has no native int64) */
    SEXP r_hist = PROTECT(allocVector(REALSXP, hist_size));
    for (int i = 0; i < hist_size; i++) {
        REAL(r_hist)[i] = (double)hist[i];
    }
    SET_VECTOR_ELT(result, 0, r_hist);

    /* hist_size — int scalar */
    SET_VECTOR_ELT(result, 1, ScalarInteger(hist_size));

    /* multisets and num_subgraphs (when requested) */
    if (keep_ms) {
        if (ms_ptr && num_sub > 0) {
            SEXP r_ms = PROTECT(allocMatrix(REALSXP, (int)num_sub, E));
            for (int64_t s = 0; s < num_sub; s++)
                for (int e = 0; e < E; e++)
                    REAL(r_ms)[e * (int)num_sub + (int)s] = ms_ptr[s * E + e];
            SET_VECTOR_ELT(result, 2, r_ms);
            UNPROTECT(1); /* r_ms */
            free(ms_ptr);
        } else {
            SET_VECTOR_ELT(result, 2, allocMatrix(REALSXP, 0, 0));
        }
        SET_VECTOR_ELT(result, 3, ScalarInteger((int)num_sub));
    }

    free(hist);
    free_dress_graph(g);
    UNPROTECT(3);
    return result;
}

#ifdef DRESS_CUDA

/* ------------------------------------------------------------------ */
/*  delta_dress_fit_mpi_cuda                                           */
/* ------------------------------------------------------------------ */
SEXP C_delta_dress_fit_mpi_cuda(SEXP n_vertices_,
                                SEXP sources_,
                                SEXP targets_,
                                SEXP weights_,
                                SEXP k_,
                                SEXP variant_,
                                SEXP max_iterations_,
                                SEXP epsilon_,
                                SEXP precompute_,
                                SEXP keep_multisets_,
                                SEXP comm_f_) {

    int N  = INTEGER(n_vertices_)[0];
    int E  = LENGTH(sources_);
    int k              = INTEGER(k_)[0];
    int variant        = INTEGER(variant_)[0];
    int max_iterations = INTEGER(max_iterations_)[0];
    double epsilon     = REAL(epsilon_)[0];
    int precompute     = INTEGER(precompute_)[0];
    int keep_ms        = INTEGER(keep_multisets_)[0];
    int comm_f         = INTEGER(comm_f_)[0];

    int *U = (int *)malloc(E * sizeof(int));
    int *V = (int *)malloc(E * sizeof(int));
    if (!U || !V) {
        free(U); free(V);
        error("delta_dress_fit_mpi_cuda: memory allocation failed");
    }
    memcpy(U, INTEGER(sources_), E * sizeof(int));
    memcpy(V, INTEGER(targets_), E * sizeof(int));

    double *W = NULL;
    if (!isNull(weights_)) {
        W = (double *)malloc(E * sizeof(double));
        if (!W) { free(U); free(V); error("delta_dress_fit_mpi_cuda: memory allocation failed"); }
        memcpy(W, REAL(weights_), E * sizeof(double));
    }

    p_dress_graph_t g = init_dress_graph(N, E, U, V, W,
                                         (dress_variant_t)variant, precompute);
    if (!g) {
        error("delta_dress_fit_mpi_cuda: init_dress_graph returned NULL");
    }

    int hist_size = 0;
    double *ms_ptr = NULL;
    int64_t num_sub = 0;
    int64_t *hist = delta_dress_fit_mpi_cuda_fcomm(g, k, max_iterations, epsilon,
                                                   &hist_size,
                                                   keep_ms,
                                                   keep_ms ? &ms_ptr : NULL,
                                                   keep_ms ? &num_sub : NULL,
                                                   comm_f);

    int n_fields = keep_ms ? 4 : 2;
    SEXP result = PROTECT(allocVector(VECSXP, n_fields));
    SEXP names  = PROTECT(allocVector(STRSXP, n_fields));

    SET_STRING_ELT(names, 0, mkChar("histogram"));
    SET_STRING_ELT(names, 1, mkChar("hist_size"));
    if (keep_ms) {
        SET_STRING_ELT(names, 2, mkChar("multisets"));
        SET_STRING_ELT(names, 3, mkChar("num_subgraphs"));
    }
    setAttrib(result, R_NamesSymbol, names);

    SEXP r_hist = PROTECT(allocVector(REALSXP, hist_size));
    for (int i = 0; i < hist_size; i++) {
        REAL(r_hist)[i] = (double)hist[i];
    }
    SET_VECTOR_ELT(result, 0, r_hist);
    SET_VECTOR_ELT(result, 1, ScalarInteger(hist_size));

    if (keep_ms) {
        if (ms_ptr && num_sub > 0) {
            SEXP r_ms = PROTECT(allocMatrix(REALSXP, (int)num_sub, E));
            for (int64_t s = 0; s < num_sub; s++)
                for (int e = 0; e < E; e++)
                    REAL(r_ms)[e * (int)num_sub + (int)s] = ms_ptr[s * E + e];
            SET_VECTOR_ELT(result, 2, r_ms);
            UNPROTECT(1);
            free(ms_ptr);
        } else {
            SET_VECTOR_ELT(result, 2, allocMatrix(REALSXP, 0, 0));
        }
        SET_VECTOR_ELT(result, 3, ScalarInteger((int)num_sub));
    }

    free(hist);
    free_dress_graph(g);
    UNPROTECT(3);
    return result;
}

#endif /* DRESS_CUDA */
#endif /* DRESS_MPI */

/* Prevent -Wempty-translation-unit when MPI/CUDA is unavailable. */
typedef int dress_mpi_r_unused_t;
