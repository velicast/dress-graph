/*
 * dress_cuda_r.c -- R bridge for the CUDA-accelerated DRESS library.
 *
 * Same API as dress_r.c but calls dress_fit_cuda / delta_dress_fit_cuda.
 * Exposed via the `cuda` environment in R:
 *
 *   library(dress)
 *   cuda$dress_fit(4, sources, targets)
 *
 * Only compiled when DRESS_CUDA is defined (requires CUDA toolkit).
 */

#ifdef DRESS_CUDA

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include "dress/dress.h"
#include "dress/delta_dress.h"
#include "dress/cuda/dress_cuda.h"

/* ------------------------------------------------------------------ */
/*  dress_fit  (CUDA)                                                  */
/* ------------------------------------------------------------------ */
SEXP C_dress_fit_cuda(SEXP n_vertices_,
                      SEXP sources_,
                      SEXP targets_,
                      SEXP weights_,
                      SEXP variant_,
                      SEXP max_iterations_,
                      SEXP epsilon_,
                      SEXP precompute_) {

    int N  = INTEGER(n_vertices_)[0];
    int E  = LENGTH(sources_);
    int variant        = INTEGER(variant_)[0];
    int max_iterations = INTEGER(max_iterations_)[0];
    double epsilon     = REAL(epsilon_)[0];
    int precompute     = INTEGER(precompute_)[0];

    int *U = (int *)malloc(E * sizeof(int));
    int *V = (int *)malloc(E * sizeof(int));
    double *W = NULL;

    if (!U || !V) {
        free(U); free(V);
        error("dress_fit_cuda: memory allocation failed");
    }

    memcpy(U, INTEGER(sources_), E * sizeof(int));
    memcpy(V, INTEGER(targets_), E * sizeof(int));

    if (weights_ != R_NilValue) {
        W = (double *)malloc(E * sizeof(double));
        if (!W) { free(U); free(V); error("dress_fit_cuda: memory allocation failed"); }
        memcpy(W, REAL(weights_), E * sizeof(double));
    }

    p_dress_graph_t g = init_dress_graph(N, E, U, V, W,
                                         (dress_variant_t)variant,
                                         precompute);
    if (!g) {
        error("dress_fit_cuda: init_dress_graph returned NULL");
    }

    int iterations = 0;
    double delta = 0.0;
    dress_fit_cuda(g, max_iterations, epsilon, &iterations, &delta);

    /* Build result list — identical structure to C_dress_fit */
    SEXP result = PROTECT(allocVector(VECSXP, 7));
    SEXP names  = PROTECT(allocVector(STRSXP, 7));

    SET_STRING_ELT(names, 0, mkChar("sources"));
    SET_STRING_ELT(names, 1, mkChar("targets"));
    SET_STRING_ELT(names, 2, mkChar("edge_dress"));
    SET_STRING_ELT(names, 3, mkChar("edge_weight"));
    SET_STRING_ELT(names, 4, mkChar("node_dress"));
    SET_STRING_ELT(names, 5, mkChar("iterations"));
    SET_STRING_ELT(names, 6, mkChar("delta"));
    setAttrib(result, R_NamesSymbol, names);

    SEXP r_sources    = PROTECT(allocVector(INTSXP,  g->E));
    SEXP r_targets    = PROTECT(allocVector(INTSXP,  g->E));
    SEXP r_edge_dress = PROTECT(allocVector(REALSXP, g->E));
    SEXP r_edge_wt    = PROTECT(allocVector(REALSXP, g->E));
    SEXP r_node_dress = PROTECT(allocVector(REALSXP, g->N));
    SEXP r_iters      = PROTECT(ScalarInteger(iterations));
    SEXP r_delta      = PROTECT(ScalarReal(delta));

    memcpy(INTEGER(r_sources),    g->U,          g->E * sizeof(int));
    memcpy(INTEGER(r_targets),    g->V,          g->E * sizeof(int));
    memcpy(REAL(r_edge_dress),    g->edge_dress, g->E * sizeof(double));
    memcpy(REAL(r_edge_wt),       g->edge_weight,g->E * sizeof(double));
    memcpy(REAL(r_node_dress),    g->node_dress, g->N * sizeof(double));

    SET_VECTOR_ELT(result, 0, r_sources);
    SET_VECTOR_ELT(result, 1, r_targets);
    SET_VECTOR_ELT(result, 2, r_edge_dress);
    SET_VECTOR_ELT(result, 3, r_edge_wt);
    SET_VECTOR_ELT(result, 4, r_node_dress);
    SET_VECTOR_ELT(result, 5, r_iters);
    SET_VECTOR_ELT(result, 6, r_delta);

    free_dress_graph(g);
    UNPROTECT(9);
    return result;
}

/* ------------------------------------------------------------------ */
/*  delta_dress_fit  (CUDA)                                            */
/* ------------------------------------------------------------------ */
SEXP C_delta_dress_fit_cuda(SEXP n_vertices_,
                            SEXP sources_,
                            SEXP targets_,
                            SEXP weights_,
                            SEXP k_,
                            SEXP variant_,
                            SEXP max_iterations_,
                            SEXP epsilon_,
                            SEXP precompute_,
                            SEXP keep_multisets_) {

    int N  = INTEGER(n_vertices_)[0];
    int E  = LENGTH(sources_);
    int k              = INTEGER(k_)[0];
    int variant        = INTEGER(variant_)[0];
    int max_iterations = INTEGER(max_iterations_)[0];
    double epsilon     = REAL(epsilon_)[0];
    int precompute     = INTEGER(precompute_)[0];
    int keep_ms        = INTEGER(keep_multisets_)[0];

    int *U = (int *)malloc(E * sizeof(int));
    int *V = (int *)malloc(E * sizeof(int));
    if (!U || !V) {
        free(U); free(V);
        error("delta_dress_fit_cuda: memory allocation failed");
    }
    memcpy(U, INTEGER(sources_), E * sizeof(int));
    memcpy(V, INTEGER(targets_), E * sizeof(int));

    double *W = NULL;
    if (!isNull(weights_)) {
        W = (double *)malloc(E * sizeof(double));
        if (!W) { free(U); free(V); error("delta_dress_fit_cuda: memory allocation failed"); }
        memcpy(W, REAL(weights_), E * sizeof(double));
    }

    p_dress_graph_t g = init_dress_graph(N, E, U, V, W,
                                         (dress_variant_t)variant, precompute);
    if (!g) {
        error("delta_dress_fit_cuda: init_dress_graph returned NULL");
    }

    int hist_size = 0;
    double *ms_ptr = NULL;
    int64_t num_sub = 0;
    int64_t *hist = delta_dress_fit_cuda(g, k, max_iterations, epsilon,
                                         &hist_size,
                                         keep_ms,
                                         keep_ms ? &ms_ptr : NULL,
                                         keep_ms ? &num_sub : NULL);

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

/* Avoid "empty translation unit" warning (-Wpedantic) when CUDA is disabled. */
typedef int dress_cuda_r_dummy_;
