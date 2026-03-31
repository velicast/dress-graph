/*
 * delta_dress_cuda_mex.c — MATLAB MEX gateway for GPU-accelerated Δ^k-DRESS.
 *
 * Same interface as delta_dress_mex.c but calls dress_delta_fit_cuda().
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
        mexErrMsgIdAndTxt("delta_dress_cuda:invalidInput",
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
    mexErrMsgIdAndTxt("delta_dress_cuda:invalidInput",
                      "%s must be a scalar integer (int32 or double).", name);
    return 0;
}

static int *to_int_array(const mxArray *arg, int expected_len, const char *name)
{
    int *out;
    mwSize n = mxGetNumberOfElements(arg);
    int i;

    if ((int)n != expected_len)
        mexErrMsgIdAndTxt("delta_dress_cuda:invalidInput",
                          "%s must have %d elements.", name, expected_len);

    out = (int *)malloc(expected_len * sizeof(int));
    if (!out)
        mexErrMsgIdAndTxt("delta_dress_cuda:malloc", "malloc failed for %s.", name);

    if (mxIsInt32(arg)) {
        memcpy(out, mxGetData(arg), expected_len * sizeof(int));
    } else if (mxIsDouble(arg) && !mxIsComplex(arg)) {
        const double *p = mxGetPr(arg);
        for (i = 0; i < expected_len; i++)
            out[i] = (int)p[i];
    } else {
        free(out);
        mexErrMsgIdAndTxt("delta_dress_cuda:invalidInput",
                          "%s must be int32 or double.", name);
    }
    return out;
}

static mxArray *build_histogram_struct(const dress_hist_pair_t *hist, int hist_size)
{
    const char *fields[] = { "value", "count" };
    mxArray *result = mxCreateStructMatrix(1, 1, 2, fields);
    mxArray *m_values = mxCreateDoubleMatrix(hist_size, 1, mxREAL);
    mxArray *m_counts = mxCreateNumericMatrix(hist_size, 1, mxINT64_CLASS, mxREAL);
    double *values = mxGetPr(m_values);
    int64_t *counts = (int64_t *)mxGetData(m_counts);
    int i;

    for (i = 0; i < hist_size; i++) {
        values[i] = hist[i].value;
        counts[i] = hist[i].count;
    }

    mxSetFieldByNumber(result, 0, 0, m_values);
    mxSetFieldByNumber(result, 0, 1, m_counts);
    return result;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    int N, E, k, variant_val, max_iterations, precompute, keep_ms;
    int n_samples = 0;
    unsigned int seed = 0;
    int compute_hist = 1;
    double epsilon;
    int    *U, *V;
    double *W = NULL;
    p_dress_graph_t g;
    int    hist_size = 0;
    dress_hist_pair_t *hist;
    double *ms_ptr = NULL;
    int64_t num_sub = 0;

    if (nrhs < 8 || nrhs > 10)
        mexErrMsgIdAndTxt("delta_dress_cuda:nrhs",
            "8 to 10 inputs required:\n"
            "  delta_dress_cuda_mex(n_vertices, sources, targets, weights, k, "
            "variant, max_iterations, epsilon, precompute, keep_multisets)");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("delta_dress_cuda:nlhs", "At most one output.");

    N = get_scalar_int(prhs[0], "n_vertices");

    E = (int)mxGetNumberOfElements(prhs[1]);
    U = to_int_array(prhs[1], E, "sources");
    V = to_int_array(prhs[2], E, "targets");

    if (!mxIsEmpty(prhs[3])) {
        W = (double *)malloc(E * sizeof(double));
        if (!W) { free(U); free(V); mexErrMsgIdAndTxt("delta_dress_cuda:oom", "malloc failed"); }
        double *src_w = mxGetPr(prhs[3]);
        for (int i = 0; i < E; i++) W[i] = src_w[i];
    }

    k              = get_scalar_int(prhs[4], "k");
    variant_val    = get_scalar_int(prhs[5], "variant");
    max_iterations = get_scalar_int(prhs[6], "max_iterations");
    epsilon        = get_scalar_double(prhs[7], "epsilon");
    precompute     = (nrhs > 8) ? get_scalar_int(prhs[8], "precompute") : 0;
    keep_ms        = (nrhs > 9) ? get_scalar_int(prhs[9], "keep_multisets") : 0;

    if (variant_val < 0 || variant_val > 3) {
        free(U); free(V); free(W);
        mexErrMsgIdAndTxt("delta_dress_cuda:invalidVariant",
                          "variant must be 0..3.");
    }
    if (k < 0) {
        free(U); free(V); free(W);
        mexErrMsgIdAndTxt("delta_dress_cuda:invalidK", "k must be >= 0.");
    }

    g = dress_init_graph(N, E, U, V, W,
                         NULL, (dress_variant_t)variant_val, precompute);
    if (!g)
        mexErrMsgIdAndTxt("delta_dress_cuda:initFailed",
                          "dress_init_graph returned NULL.");

    hist = dress_delta_fit_cuda(g, k, max_iterations, epsilon,
                                n_samples, seed,
                                compute_hist ? &hist_size : NULL,
                                keep_ms,
                                keep_ms ? &ms_ptr : NULL,
                                &num_sub);

    int n_fields = keep_ms ? 3 : 1;
    const char *fields_basic[]  = { "histogram" };
    const char *fields_full[]   = { "histogram", "multisets", "num_subgraphs" };
    plhs[0] = mxCreateStructMatrix(1, 1, n_fields,
                                   keep_ms ? fields_full : fields_basic);

    mxSetFieldByNumber(plhs[0], 0, 0, build_histogram_struct(hist, hist_size));

    if (keep_ms && ms_ptr) {
        mxArray *m_ms = mxCreateDoubleMatrix((mwSize)num_sub, (mwSize)E, mxREAL);
        {
            double *dst = mxGetPr(m_ms);
            int64_t s;
            int e;
            for (s = 0; s < num_sub; s++)
                for (e = 0; e < E; e++)
                    dst[e * (mwSize)num_sub + s] = ms_ptr[s * E + e];
        }
        mxSetFieldByNumber(plhs[0], 0, 1, m_ms);

        mxArray *m_nsub = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
        ((int *)mxGetData(m_nsub))[0] = (int)num_sub;
        mxSetFieldByNumber(plhs[0], 0, 2, m_nsub);

        free(ms_ptr);
    } else if (keep_ms) {
        mxSetFieldByNumber(plhs[0], 0, 1, mxCreateDoubleMatrix(0, 0, mxREAL));
        mxArray *m_nsub = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
        ((int *)mxGetData(m_nsub))[0] = 0;
        mxSetFieldByNumber(plhs[0], 0, 2, m_nsub);
    }

    free(hist);
    dress_free_graph(g);
}
