/*
 * dress_init_mex.c — MEX gateway: create a persistent DRESS graph.
 *
 * Usage:
 *   ptr = dress_init_mex(n_vertices, sources, targets, weights, variant, precompute)
 *
 * Returns a uint64 scalar holding the opaque C pointer.
 */

#include "mex.h"
#if defined(__has_include)
#  if __has_include("matrix.h")
#    include "matrix.h"
#  endif
#else
#  include "matrix.h"
#endif
#include "dress/dress.h"

#include <string.h>
#include <stdlib.h>
#include <stdint.h>

static int get_scalar_int(const mxArray *arg, const char *name)
{
    if (mxIsInt32(arg) && mxGetNumberOfElements(arg) == 1)
        return ((int *)mxGetData(arg))[0];
    if (mxIsDouble(arg) && !mxIsComplex(arg) && mxGetNumberOfElements(arg) == 1)
        return (int)mxGetScalar(arg);
    mexErrMsgIdAndTxt("dress:invalidInput",
                      "%s must be a scalar integer.", name);
    return 0;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    int N, E, variant_val, precompute;
    int *U, *V;
    double *W = NULL;
    int i;

    if (nrhs != 6)
        mexErrMsgIdAndTxt("dress:nrhs",
            "Six inputs required: dress_init_mex(n, sources, targets, weights, variant, precompute)");

    N           = get_scalar_int(prhs[0], "n_vertices");
    variant_val = get_scalar_int(prhs[4], "variant");
    precompute  = get_scalar_int(prhs[5], "precompute");

    E = (int)mxGetNumberOfElements(prhs[1]);

    /* Allocate copies (dress_init_graph takes ownership). */
    U = (int *)malloc(E * sizeof(int));
    V = (int *)malloc(E * sizeof(int));
    if (!U || !V) {
        free(U); free(V);
        mexErrMsgIdAndTxt("dress:malloc", "malloc failed.");
    }

    if (mxIsInt32(prhs[1])) {
        memcpy(U, mxGetData(prhs[1]), E * sizeof(int));
    } else {
        const double *p = mxGetPr(prhs[1]);
        for (i = 0; i < E; i++) U[i] = (int)p[i];
    }
    if (mxIsInt32(prhs[2])) {
        memcpy(V, mxGetData(prhs[2]), E * sizeof(int));
    } else {
        const double *p = mxGetPr(prhs[2]);
        for (i = 0; i < E; i++) V[i] = (int)p[i];
    }

    if (!mxIsEmpty(prhs[3])) {
        W = (double *)malloc(E * sizeof(double));
        if (!W) { free(U); free(V); mexErrMsgIdAndTxt("dress:malloc", "malloc failed."); }
        memcpy(W, mxGetPr(prhs[3]), E * sizeof(double));
    }

    p_dress_graph_t g = dress_init_graph(N, E, U, V, W,
                                         NULL, (dress_variant_t)variant_val,
                                         precompute);
    if (!g) {
        mexErrMsgIdAndTxt("dress:initFailed", "dress_init_graph returned NULL.");
    }

    /* Return pointer as uint64 */
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t *)mxGetData(plhs[0])) = (uint64_t)(uintptr_t)g;
}
