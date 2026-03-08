/*
 * delta_dress_cuda_mex.c — MATLAB MEX gateway for GPU-accelerated Δ^k-DRESS.
 *
 * Same interface as delta_dress_mex.c but calls delta_dress_fit_cuda().
 */

#include "mex.h"
#if !defined(__OCTAVE__) && !defined(OCTAVE_MEX_FILE)
#include "matrix.h"
#endif
#include "dress/dress.h"
#include "dress/delta_dress.h"
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

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    int N, E, k, variant_val, max_iterations, precompute, keep_ms;
    double epsilon;
    int    *U, *V;
    double *W = NULL;
    p_dress_graph_t g;
    int    hist_size = 0;
    int64_t *hist;
    double *ms_ptr = NULL;
    int64_t num_sub = 0;
    mxArray *m_hist, *m_hsize;

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

    g = init_dress_graph(N, E, U, V, W,
                         (dress_variant_t)variant_val, precompute);
    if (!g)
        mexErrMsgIdAndTxt("delta_dress_cuda:initFailed",
                          "init_dress_graph returned NULL.");

    hist = delta_dress_fit_cuda(g, k, max_iterations, epsilon, &hist_size,
                                keep_ms,
                                keep_ms ? &ms_ptr : NULL,
                                keep_ms ? &num_sub : NULL);

    int n_fields = keep_ms ? 4 : 2;
    const char *fields_basic[]  = { "histogram", "hist_size" };
    const char *fields_full[]   = { "histogram", "hist_size", "multisets", "num_subgraphs" };
    plhs[0] = mxCreateStructMatrix(1, 1, n_fields,
                                   keep_ms ? fields_full : fields_basic);

    m_hist = mxCreateDoubleMatrix(hist_size, 1, mxREAL);
    {
        double *dst = mxGetPr(m_hist);
        int i;
        for (i = 0; i < hist_size; i++)
            dst[i] = (double)hist[i];
    }
    mxSetFieldByNumber(plhs[0], 0, 0, m_hist);

    m_hsize = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    ((int *)mxGetData(m_hsize))[0] = hist_size;
    mxSetFieldByNumber(plhs[0], 0, 1, m_hsize);

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
        mxSetFieldByNumber(plhs[0], 0, 2, m_ms);

        mxArray *m_nsub = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
        ((int *)mxGetData(m_nsub))[0] = (int)num_sub;
        mxSetFieldByNumber(plhs[0], 0, 3, m_nsub);

        free(ms_ptr);
    } else if (keep_ms) {
        mxSetFieldByNumber(plhs[0], 0, 2, mxCreateDoubleMatrix(0, 0, mxREAL));
        mxArray *m_nsub = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
        ((int *)mxGetData(m_nsub))[0] = 0;
        mxSetFieldByNumber(plhs[0], 0, 3, m_nsub);
    }

    free(hist);
    free_dress_graph(g);
}
