/*
 * dress_r.c -- R bridge for the DRESS library
 *
 * Provides .Call-compatible C functions that wrap the DRESS C API.
 */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include "dress/dress.h"

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
    fit(g, max_iterations, epsilon, &iterations, &delta);

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
    return ScalarString(mkChar("0.1.1"));
}

/* ------------------------------------------------------------------ */
/*  Registration table                                                 */
/* ------------------------------------------------------------------ */
static const R_CallMethodDef callMethods[] = {
    {"C_dress_fit",     (DL_FUNC) &C_dress_fit,     8},
    {"C_dress_version", (DL_FUNC) &C_dress_version, 0},
    {NULL, NULL, 0}
};

void R_init_dress_graph(DllInfo *dll) {
    R_registerRoutines(dll, NULL, callMethods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
