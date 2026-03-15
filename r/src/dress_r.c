/*
 * dress_r.c -- R bridge for the DRESS library
 *
 * Provides .Call-compatible C functions that wrap the DRESS C API.
 */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include "dress/dress.h"
#include "dress/delta_dress.h"

/* ------------------------------------------------------------------ */
/*  dress_fit                                                          */
/* ------------------------------------------------------------------ */
SEXP C_dress_fit(SEXP n_vertices_,
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

    /* Allocate copies of sources/targets/weights with malloc
       (init_dress_graph takes ownership and frees them). */
    int *U = (int *)malloc(E * sizeof(int));
    int *V = (int *)malloc(E * sizeof(int));
    double *W = NULL;

    if (!U || !V) {
        free(U); free(V);
        error("dress_fit: memory allocation failed");
    }

    memcpy(U, INTEGER(sources_), E * sizeof(int));
    memcpy(V, INTEGER(targets_), E * sizeof(int));

    if (weights_ != R_NilValue) {
        W = (double *)malloc(E * sizeof(double));
        if (!W) { free(U); free(V); error("dress_fit: memory allocation failed"); }
        memcpy(W, REAL(weights_), E * sizeof(double));
    }

    p_dress_graph_t g = init_dress_graph(N, E, U, V, W,
                                         (dress_variant_t)variant,
                                         precompute);
    if (!g) {
        error("dress_fit: init_dress_graph returned NULL");
    }

    int iterations = 0;
    double delta = 0.0;
    dress_fit(g, max_iterations, epsilon, &iterations, &delta);

    /* Build result list */
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
/*  dress_version                                                      */
/* ------------------------------------------------------------------ */
SEXP C_dress_version(void) {
    return ScalarString(mkChar("0.5.2"));
}

/* ------------------------------------------------------------------ */
/*  delta_dress_fit                                                    */
/* ------------------------------------------------------------------ */
SEXP C_delta_dress_fit(SEXP n_vertices_,
                       SEXP sources_,
                       SEXP targets_,
                       SEXP weights_,
                       SEXP k_,
                       SEXP variant_,
                       SEXP max_iterations_,
                       SEXP epsilon_,
                       SEXP precompute_,
                       SEXP keep_multisets_,
                       SEXP offset_,
                       SEXP stride_) {

    int N  = INTEGER(n_vertices_)[0];
    int E  = LENGTH(sources_);
    int k              = INTEGER(k_)[0];
    int variant        = INTEGER(variant_)[0];
    int max_iterations = INTEGER(max_iterations_)[0];
    double epsilon     = REAL(epsilon_)[0];
    int precompute     = INTEGER(precompute_)[0];
    int keep_ms        = INTEGER(keep_multisets_)[0];
    int offset         = INTEGER(offset_)[0];
    int stride         = INTEGER(stride_)[0];

    /* Allocate copies (init_dress_graph takes ownership). */
    int *U = (int *)malloc(E * sizeof(int));
    int *V = (int *)malloc(E * sizeof(int));
    if (!U || !V) {
        free(U); free(V);
        error("delta_dress_fit: memory allocation failed");
    }
    memcpy(U, INTEGER(sources_), E * sizeof(int));
    memcpy(V, INTEGER(targets_), E * sizeof(int));

    double *W = NULL;
    if (!isNull(weights_)) {
        W = (double *)malloc(E * sizeof(double));
        if (!W) { free(U); free(V); error("delta_dress_fit: memory allocation failed"); }
        memcpy(W, REAL(weights_), E * sizeof(double));
    }

    p_dress_graph_t g = init_dress_graph(N, E, U, V, W,
                                         (dress_variant_t)variant, precompute);
    if (!g) {
        error("delta_dress_fit: init_dress_graph returned NULL");
    }

    int hist_size = 0;
    double *ms_ptr = NULL;
    int64_t num_sub = 0;
    int64_t *hist = delta_dress_fit_strided(g, k, max_iterations, epsilon,
                              &hist_size,
                              keep_ms,
                              keep_ms ? &ms_ptr : NULL,
                              keep_ms ? &num_sub : NULL,
                              offset, stride);

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
            /* R matrix is column-major: matrix[row, col] = data[col * nrow + row]
               C multisets are row-major: ms_ptr[s * E + e]
               R matrix(num_sub, E): result[s, e] = data[(e-1)*num_sub + (s-1)]
               So we transpose during copy. */
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
/* CUDA bridge (defined in dress_cuda_r.c) */
extern SEXP C_dress_fit_cuda(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP C_delta_dress_fit_cuda(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP C_dress_fit_cuda_obj(SEXP, SEXP, SEXP);
#endif

#ifdef DRESS_MPI
/* MPI bridge (defined in dress_mpi_r.c) */
extern SEXP C_delta_dress_fit_mpi(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
#ifdef DRESS_CUDA
extern SEXP C_delta_dress_fit_mpi_cuda(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
#endif
#endif

/* ------------------------------------------------------------------ */
/*  Persistent DRESS object (external pointer)                         */
/* ------------------------------------------------------------------ */

static void dress_graph_finalizer(SEXP ptr) {
    p_dress_graph_t g = (p_dress_graph_t) R_ExternalPtrAddr(ptr);
    if (g) {
        free_dress_graph(g);
        R_ClearExternalPtr(ptr);
    }
}

/* Create and return an external pointer wrapping a C dress_graph_t. */
SEXP C_dress_init(SEXP n_vertices_,
                  SEXP sources_,
                  SEXP targets_,
                  SEXP weights_,
                  SEXP variant_,
                  SEXP precompute_) {

    int N  = INTEGER(n_vertices_)[0];
    int E  = LENGTH(sources_);
    int variant    = INTEGER(variant_)[0];
    int precompute = INTEGER(precompute_)[0];

    int *U = (int *)malloc(E * sizeof(int));
    int *V = (int *)malloc(E * sizeof(int));
    double *W = NULL;

    if (!U || !V) {
        free(U); free(V);
        error("dress_init: memory allocation failed");
    }
    memcpy(U, INTEGER(sources_), E * sizeof(int));
    memcpy(V, INTEGER(targets_), E * sizeof(int));

    if (weights_ != R_NilValue) {
        W = (double *)malloc(E * sizeof(double));
        if (!W) { free(U); free(V); error("dress_init: memory allocation failed"); }
        memcpy(W, REAL(weights_), E * sizeof(double));
    }

    p_dress_graph_t g = init_dress_graph(N, E, U, V, W,
                                         (dress_variant_t)variant,
                                         precompute);
    if (!g) {
        error("dress_init: init_dress_graph returned NULL");
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(g, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, dress_graph_finalizer, TRUE);
    UNPROTECT(1);
    return ptr;
}

/* Fit an already-initialized graph. Returns list(iterations, delta). */
SEXP C_dress_fit_obj(SEXP ptr_,
                     SEXP max_iterations_,
                     SEXP epsilon_) {

    p_dress_graph_t g = (p_dress_graph_t) R_ExternalPtrAddr(ptr_);
    if (!g) error("dress_fit_obj: graph already closed");

    int max_iterations = INTEGER(max_iterations_)[0];
    double epsilon     = REAL(epsilon_)[0];

    int iterations = 0;
    double delta = 0.0;
    dress_fit(g, max_iterations, epsilon, &iterations, &delta);

    SEXP result = PROTECT(allocVector(VECSXP, 2));
    SEXP names  = PROTECT(allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, mkChar("iterations"));
    SET_STRING_ELT(names, 1, mkChar("delta"));
    setAttrib(result, R_NamesSymbol, names);
    SET_VECTOR_ELT(result, 0, ScalarInteger(iterations));
    SET_VECTOR_ELT(result, 1, ScalarReal(delta));
    UNPROTECT(2);
    return result;
}

/* Query DRESS value for an edge (existing or virtual). */
SEXP C_dress_get_obj(SEXP ptr_,
                     SEXP u_,
                     SEXP v_,
                     SEXP max_iterations_,
                     SEXP epsilon_,
                     SEXP edge_weight_) {

    p_dress_graph_t g = (p_dress_graph_t) R_ExternalPtrAddr(ptr_);
    if (!g) error("dress_get_obj: graph already closed");

    int u = INTEGER(u_)[0];
    int v = INTEGER(v_)[0];
    int max_iterations = INTEGER(max_iterations_)[0];
    double epsilon     = REAL(epsilon_)[0];
    double edge_weight = REAL(edge_weight_)[0];

    double val = dress_get(g, u, v, max_iterations, epsilon, edge_weight);
    return ScalarReal(val);
}

/* Extract current results without freeing. */
SEXP C_dress_result(SEXP ptr_) {
    p_dress_graph_t g = (p_dress_graph_t) R_ExternalPtrAddr(ptr_);
    if (!g) error("dress_result: graph already closed");

    SEXP result = PROTECT(allocVector(VECSXP, 5));
    SEXP names  = PROTECT(allocVector(STRSXP, 5));

    SET_STRING_ELT(names, 0, mkChar("sources"));
    SET_STRING_ELT(names, 1, mkChar("targets"));
    SET_STRING_ELT(names, 2, mkChar("edge_dress"));
    SET_STRING_ELT(names, 3, mkChar("edge_weight"));
    SET_STRING_ELT(names, 4, mkChar("node_dress"));
    setAttrib(result, R_NamesSymbol, names);

    SEXP r_sources    = PROTECT(allocVector(INTSXP,  g->E));
    SEXP r_targets    = PROTECT(allocVector(INTSXP,  g->E));
    SEXP r_edge_dress = PROTECT(allocVector(REALSXP, g->E));
    SEXP r_edge_wt    = PROTECT(allocVector(REALSXP, g->E));
    SEXP r_node_dress = PROTECT(allocVector(REALSXP, g->N));

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

    UNPROTECT(7);
    return result;
}

/* Explicitly free the underlying C graph. */
SEXP C_dress_close(SEXP ptr_) {
    p_dress_graph_t g = (p_dress_graph_t) R_ExternalPtrAddr(ptr_);
    if (g) {
        free_dress_graph(g);
        R_ClearExternalPtr(ptr_);
    }
    return R_NilValue;
}

/* ------------------------------------------------------------------ */
/*  Registration table                                                 */
/* ------------------------------------------------------------------ */
static const R_CallMethodDef callMethods[] = {
    {"C_dress_fit",            (DL_FUNC) &C_dress_fit,            8},
    {"C_delta_dress_fit",      (DL_FUNC) &C_delta_dress_fit,      12},
    {"C_dress_version",        (DL_FUNC) &C_dress_version,        0},
    {"C_dress_init",           (DL_FUNC) &C_dress_init,           6},
    {"C_dress_fit_obj",        (DL_FUNC) &C_dress_fit_obj,        3},
    {"C_dress_get_obj",        (DL_FUNC) &C_dress_get_obj,        6},
    {"C_dress_result",         (DL_FUNC) &C_dress_result,         1},
    {"C_dress_close",          (DL_FUNC) &C_dress_close,          1},
#ifdef DRESS_CUDA
    {"C_dress_fit_cuda",       (DL_FUNC) &C_dress_fit_cuda,       8},
    {"C_delta_dress_fit_cuda", (DL_FUNC) &C_delta_dress_fit_cuda, 12},
    {"C_dress_fit_cuda_obj",   (DL_FUNC) &C_dress_fit_cuda_obj,   3},
#endif
#ifdef DRESS_MPI
    {"C_delta_dress_fit_mpi",  (DL_FUNC) &C_delta_dress_fit_mpi,  11},
#ifdef DRESS_CUDA
    {"C_delta_dress_fit_mpi_cuda", (DL_FUNC) &C_delta_dress_fit_mpi_cuda, 11},
#endif
#endif
    {NULL, NULL, 0}
};

void R_init_dress_graph(DllInfo *dll) {
    R_registerRoutines(dll, NULL, callMethods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
