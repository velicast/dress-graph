/*
 * dress_cuda_mex.c — MATLAB MEX gateway for GPU-accelerated DRESS.
 *
 * Same interface as dress_mex.c but calls dress_fit_cuda().
 *
 * Compile:
 *   mex -O -I../libdress/include CFLAGS="$CFLAGS -fopenmp" ...
 *       LDFLAGS="$LDFLAGS -fopenmp -lcudart" ...
 *       dress_cuda_mex.c ../libdress/src/dress.c -ldress_cuda -lm
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
#include "dress/cuda/dress_cuda.h"

#include <string.h>
#include <stdlib.h>

static double get_scalar_double(const mxArray *arg, const char *name)
{
    if (!mxIsDouble(arg) || mxIsComplex(arg) || mxGetNumberOfElements(arg) != 1)
        mexErrMsgIdAndTxt("dress_cuda:invalidInput",
                          "%s must be a real scalar double.", name);
    return mxGetScalar(arg);
}

static int get_scalar_int(const mxArray *arg, const char *name)
{
    double v;
    if (mxIsInt32(arg) && mxGetNumberOfElements(arg) == 1)
        return ((int *)mxGetData(arg))[0];
    if (mxIsDouble(arg) && !mxIsComplex(arg) && mxGetNumberOfElements(arg) == 1) {
        v = mxGetScalar(arg);
        return (int)v;
    }
    mexErrMsgIdAndTxt("dress_cuda:invalidInput",
                      "%s must be a scalar integer (int32 or double).", name);
    return 0;
}

static int *to_int_array(const mxArray *arg, int expected_len, const char *name)
{
    int *out;
    mwSize n = mxGetNumberOfElements(arg);
    int i;

    if ((int)n != expected_len)
        mexErrMsgIdAndTxt("dress_cuda:invalidInput",
                          "%s must have %d elements.", name, expected_len);

    out = (int *)malloc(expected_len * sizeof(int));
    if (!out)
        mexErrMsgIdAndTxt("dress_cuda:malloc", "malloc failed for %s.", name);

    if (mxIsInt32(arg)) {
        memcpy(out, mxGetData(arg), expected_len * sizeof(int));
    } else if (mxIsDouble(arg) && !mxIsComplex(arg)) {
        const double *p = mxGetPr(arg);
        for (i = 0; i < expected_len; i++)
            out[i] = (int)p[i];
    } else {
        free(out);
        mexErrMsgIdAndTxt("dress_cuda:invalidInput",
                          "%s must be int32 or double.", name);
    }
    return out;
}

static double *to_double_array(const mxArray *arg, int expected_len,
                               const char *name)
{
    double *out;
    mwSize n = mxGetNumberOfElements(arg);

    if (!mxIsDouble(arg) || mxIsComplex(arg))
        mexErrMsgIdAndTxt("dress_cuda:invalidInput",
                          "%s must be a real double array.", name);
    if ((int)n != expected_len)
        mexErrMsgIdAndTxt("dress_cuda:invalidInput",
                          "%s must have %d elements.", name, expected_len);

    out = (double *)malloc(expected_len * sizeof(double));
    if (!out)
        mexErrMsgIdAndTxt("dress_cuda:malloc", "malloc failed for %s.", name);
    memcpy(out, mxGetPr(arg), expected_len * sizeof(double));
    return out;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    int N, E, variant_val, max_iterations, precompute;
    double epsilon;
    int    *U, *V;
    double *W = NULL;
    p_dress_graph_t g;
    int    iterations = 0;
    double delta      = 0.0;

    const char *field_names[] = {
        "sources", "targets", "edge_dress", "edge_weight",
        "vertex_dress", "iterations", "delta"
    };

    mxArray *m_src, *m_dst, *m_dress, *m_weight, *m_node, *m_iters, *m_delta;

    if (nrhs != 8)
        mexErrMsgIdAndTxt("dress_cuda:nrhs",
            "Eight inputs required:\n"
            "  dress_cuda_mex(n_vertices, sources, targets, weights, "
            "variant, max_iterations, epsilon, precompute)");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("dress_cuda:nlhs", "At most one output.");

    N              = get_scalar_int(prhs[0], "n_vertices");
    variant_val    = get_scalar_int(prhs[4], "variant");
    max_iterations = get_scalar_int(prhs[5], "max_iterations");
    epsilon        = get_scalar_double(prhs[6], "epsilon");
    precompute     = get_scalar_int(prhs[7], "precompute");

    E = (int)mxGetNumberOfElements(prhs[1]);
    U = to_int_array(prhs[1], E, "sources");
    V = to_int_array(prhs[2], E, "targets");

    if (!mxIsEmpty(prhs[3])) {
        W = to_double_array(prhs[3], E, "weights");
    }

    if (variant_val < 0 || variant_val > 3) {
        free(U); free(V); free(W);
        mexErrMsgIdAndTxt("dress_cuda:invalidVariant",
                          "variant must be 0..3.");
    }

    g = dress_init_graph(N, E, U, V, W,
                         NULL, (dress_variant_t)variant_val,
                         precompute);
    if (!g)
        mexErrMsgIdAndTxt("dress_cuda:initFailed",
                          "dress_init_graph returned NULL.");

    dress_fit_cuda(g, max_iterations, epsilon, &iterations, &delta);

    plhs[0] = mxCreateStructMatrix(1, 1, 7, field_names);

    m_src = mxCreateNumericMatrix(E, 1, mxINT32_CLASS, mxREAL);
    memcpy(mxGetData(m_src), g->U, E * sizeof(int));
    mxSetFieldByNumber(plhs[0], 0, 0, m_src);

    m_dst = mxCreateNumericMatrix(E, 1, mxINT32_CLASS, mxREAL);
    memcpy(mxGetData(m_dst), g->V, E * sizeof(int));
    mxSetFieldByNumber(plhs[0], 0, 1, m_dst);

    m_dress = mxCreateDoubleMatrix(E, 1, mxREAL);
    memcpy(mxGetPr(m_dress), g->edge_dress, E * sizeof(double));
    mxSetFieldByNumber(plhs[0], 0, 2, m_dress);

    m_weight = mxCreateDoubleMatrix(E, 1, mxREAL);
    memcpy(mxGetPr(m_weight), g->edge_weight, E * sizeof(double));
    mxSetFieldByNumber(plhs[0], 0, 3, m_weight);

    m_node = mxCreateDoubleMatrix(N, 1, mxREAL);
    memcpy(mxGetPr(m_node), g->vertex_dress, N * sizeof(double));
    mxSetFieldByNumber(plhs[0], 0, 4, m_node);

    m_iters = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    ((int *)mxGetData(m_iters))[0] = iterations;
    mxSetFieldByNumber(plhs[0], 0, 5, m_iters);

    m_delta = mxCreateDoubleScalar(delta);
    mxSetFieldByNumber(plhs[0], 0, 6, m_delta);

    dress_free_graph(g);
}
