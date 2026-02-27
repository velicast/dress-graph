/*
 * nabla_dress_mex.c — MATLAB MEX gateway for nabla-k-DRESS.
 *
 * Compile from the matlab/ directory:
 *
 *   mex -O -I../libdress/include CFLAGS="$CFLAGS -fopenmp" LDFLAGS="$LDFLAGS -fopenmp" ...
 *       nabla_dress_mex.c ../libdress/src/dress.c ../libdress/src/nabla_dress.c -lm
 *
 * Usage in MATLAB:
 *
 *   result = nabla_dress_mex(n_vertices, sources, targets, weights, ...
 *                            k, nabla_weight, variant, max_iterations, ...
 *                            epsilon, precompute);
 *
 *   result is a struct with fields:
 *     .histogram  — double [hist_size x 1]  bin counts (double for MATLAB compat)
 *     .hist_size  — int32 scalar            number of bins
 */

#include "mex.h"
#if !defined(__OCTAVE__) && !defined(OCTAVE_MEX_FILE)
#include "matrix.h"
#endif
#include "dress/dress.h"
#include "dress/nabla_dress.h"

#include <string.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/*  Helper: read a scalar double from a MATLAB argument                */
/* ------------------------------------------------------------------ */
static double get_scalar_double(const mxArray *arg, const char *name)
{
    if (!mxIsDouble(arg) || mxIsComplex(arg) || mxGetNumberOfElements(arg) != 1)
        mexErrMsgIdAndTxt("nabla_dress:invalidInput",
                          "%s must be a real scalar double.", name);
    return mxGetScalar(arg);
}

/* ------------------------------------------------------------------ */
/*  Helper: read a scalar int32 from a MATLAB argument                 */
/* ------------------------------------------------------------------ */
static int get_scalar_int(const mxArray *arg, const char *name)
{
    double v;
    if (mxIsInt32(arg) && mxGetNumberOfElements(arg) == 1)
        return ((int *)mxGetData(arg))[0];
    if (mxIsDouble(arg) && !mxIsComplex(arg) && mxGetNumberOfElements(arg) == 1) {
        v = mxGetScalar(arg);
        return (int)v;
    }
    mexErrMsgIdAndTxt("nabla_dress:invalidInput",
                      "%s must be a scalar integer (int32 or double).", name);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Helper: copy a MATLAB numeric vector to a malloc'd int array       */
/* ------------------------------------------------------------------ */
static int *to_int_array(const mxArray *arg, int expected_len, const char *name)
{
    int *out;
    mwSize n = mxGetNumberOfElements(arg);
    int i;

    if ((int)n != expected_len)
        mexErrMsgIdAndTxt("nabla_dress:invalidInput",
                          "%s must have %d elements.", name, expected_len);

    out = (int *)malloc(expected_len * sizeof(int));
    if (!out)
        mexErrMsgIdAndTxt("nabla_dress:malloc", "malloc failed for %s.", name);

    if (mxIsInt32(arg)) {
        memcpy(out, mxGetData(arg), expected_len * sizeof(int));
    } else if (mxIsDouble(arg) && !mxIsComplex(arg)) {
        const double *p = mxGetPr(arg);
        for (i = 0; i < expected_len; i++)
            out[i] = (int)p[i];
    } else {
        free(out);
        mexErrMsgIdAndTxt("nabla_dress:invalidInput",
                          "%s must be int32 or double.", name);
    }
    return out;
}

/* ------------------------------------------------------------------ */
/*  MEX gateway                                                        */
/* ------------------------------------------------------------------ */

/*
 * MATLAB signature:
 *   result = nabla_dress_mex(n_vertices, sources, targets, weights, ...
 *                            k, nabla_weight, variant, max_iterations, ...
 *                            epsilon, precompute)
 *
 * Inputs:
 *   n_vertices       — scalar int: number of vertices
 *   sources          — int32 or double [E x 1]: edge sources (0-based)
 *   targets          — int32 or double [E x 1]: edge targets (0-based)
 *   weights          — double [E x 1] or []: edge weights
 *   k                — scalar int: vertices to individualize (0 = original)
 *   nabla_weight     — scalar double: multiplicative factor for marked edges
 *   variant          — scalar int: 0..3
 *   max_iterations   — scalar int: max fitting iterations
 *   epsilon          — scalar double: convergence / bin width
 *   precompute       — scalar int: 0 or 1
 *
 * Output:
 *   result — struct with .histogram (double [hist_size x 1]) and
 *            .hist_size (int32 scalar).
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    int N, E, k, variant_val, max_iterations, precompute;
    double epsilon, nabla_weight;
    int    *U, *V;
    double *W = NULL;
    p_dress_graph_t g;
    int    hist_size = 0;
    int64_t *hist;

    const char *field_names[] = { "histogram", "hist_size" };
    mxArray *m_hist, *m_hsize;

    /* ---- argument count check ---- */
    if (nrhs != 10)
        mexErrMsgIdAndTxt("nabla_dress:nrhs",
            "10 inputs required:\n"
            "  nabla_dress_mex(n_vertices, sources, targets, weights, k, "
            "nabla_weight, variant, max_iterations, epsilon, precompute)");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("nabla_dress:nlhs", "At most one output.");

    /* ---- unpack scalars ---- */
    N              = get_scalar_int(prhs[0], "n_vertices");

    /* ---- unpack edge arrays ---- */
    E = (int)mxGetNumberOfElements(prhs[1]);
    U = to_int_array(prhs[1], E, "sources");
    V = to_int_array(prhs[2], E, "targets");

    /* ---- unpack weights (prhs[3], may be empty) ---- */
    if (!mxIsEmpty(prhs[3])) {
        W = (double *)malloc(E * sizeof(double));
        if (!W) { free(U); free(V); mexErrMsgIdAndTxt("nabla_dress:oom", "malloc failed"); }
        double *src_w = mxGetPr(prhs[3]);
        for (int i = 0; i < E; i++) W[i] = src_w[i];
    }

    k              = get_scalar_int(prhs[4], "k");
    nabla_weight   = get_scalar_double(prhs[5], "nabla_weight");
    variant_val    = get_scalar_int(prhs[6], "variant");
    max_iterations = get_scalar_int(prhs[7], "max_iterations");
    epsilon        = get_scalar_double(prhs[8], "epsilon");
    precompute     = get_scalar_int(prhs[9], "precompute");

    /* ---- validate ---- */
    if (variant_val < 0 || variant_val > 3) {
        free(U); free(V); free(W);
        mexErrMsgIdAndTxt("nabla_dress:invalidVariant",
                          "variant must be 0..3.");
    }
    if (k < 0) {
        free(U); free(V); free(W);
        mexErrMsgIdAndTxt("nabla_dress:invalidK", "k must be >= 0.");
    }

    /* ---- build graph (takes ownership of U, V, W) ---- */
    g = init_dress_graph(N, E, U, V, W,
                         (dress_variant_t)variant_val, precompute);
    if (!g)
        mexErrMsgIdAndTxt("nabla_dress:initFailed",
                          "init_dress_graph returned NULL.");

    /* ---- compute nabla-k-dress ---- */
    hist = nabla_fit(g, k, max_iterations, epsilon, nabla_weight,
                     &hist_size, 0, NULL, NULL);

    /* ---- pack output struct ---- */
    plhs[0] = mxCreateStructMatrix(1, 1, 2, field_names);

    /* histogram — double [hist_size x 1]  (MATLAB has no int64 MEX helper) */
    m_hist = mxCreateDoubleMatrix(hist_size, 1, mxREAL);
    {
        double *dst = mxGetPr(m_hist);
        int i;
        for (i = 0; i < hist_size; i++)
            dst[i] = (double)hist[i];
    }
    mxSetFieldByNumber(plhs[0], 0, 0, m_hist);

    /* hist_size — int32 scalar */
    m_hsize = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    ((int *)mxGetData(m_hsize))[0] = hist_size;
    mxSetFieldByNumber(plhs[0], 0, 1, m_hsize);

    /* ---- cleanup ---- */
    free(hist);
    free_dress_graph(g);
}
