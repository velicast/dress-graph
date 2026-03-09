/*
 * delta_dress_mex.c — MATLAB MEX gateway for delta-k-DRESS.
 *
 * Compile from the matlab/ directory:
 *
 *   mex -O -I../libdress/include CFLAGS="$CFLAGS -fopenmp" LDFLAGS="$LDFLAGS -fopenmp" ...
 *       delta_dress_mex.c ../libdress/src/dress.c ../libdress/src/delta_dress.c -lm
 *
 * Usage in MATLAB:
 *
 *   result = delta_dress_mex(n_vertices, sources, targets, ...
 *                            k, variant, max_iterations, epsilon, precompute, ...
 *                            keep_multisets);
 *
 *   result is a struct with fields:
 *     .histogram      — double [hist_size x 1]  bin counts (double for MATLAB compat)
 *     .hist_size      — int32 scalar            number of bins
 *     .multisets      — double [C(N,k) x E]     per-subgraph edge values (NaN = removed)
 *                       (only present when keep_multisets is nonzero)
 *     .num_subgraphs  — int32 scalar            C(N,k)
 *                       (only present when keep_multisets is nonzero)
 */

#include "mex.h"
#if !defined(__OCTAVE__) && !defined(OCTAVE_MEX_FILE)
#include "matrix.h"
#endif
#include "dress/dress.h"
#include "dress/delta_dress.h"

#include <string.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/*  Helper: read a scalar double from a MATLAB argument                */
/* ------------------------------------------------------------------ */
static double get_scalar_double(const mxArray *arg, const char *name)
{
    if (!mxIsDouble(arg) || mxIsComplex(arg) || mxGetNumberOfElements(arg) != 1)
        mexErrMsgIdAndTxt("delta_dress:invalidInput",
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
    mexErrMsgIdAndTxt("delta_dress:invalidInput",
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
        mexErrMsgIdAndTxt("delta_dress:invalidInput",
                          "%s must have %d elements.", name, expected_len);

    out = (int *)malloc(expected_len * sizeof(int));
    if (!out)
        mexErrMsgIdAndTxt("delta_dress:malloc", "malloc failed for %s.", name);

    if (mxIsInt32(arg)) {
        memcpy(out, mxGetData(arg), expected_len * sizeof(int));
    } else if (mxIsDouble(arg) && !mxIsComplex(arg)) {
        const double *p = mxGetPr(arg);
        for (i = 0; i < expected_len; i++)
            out[i] = (int)p[i];
    } else {
        free(out);
        mexErrMsgIdAndTxt("delta_dress:invalidInput",
                          "%s must be int32 or double.", name);
    }
    return out;
}

/* ------------------------------------------------------------------ */
/*  MEX gateway                                                        */
/* ------------------------------------------------------------------ */

/*
 * MATLAB signature:
 *   result = delta_dress_mex(n_vertices, sources, targets, ...
 *                            weights, k, variant, max_iterations, epsilon, ...
 *                            precompute, keep_multisets)
 *
 * Inputs:
 *   n_vertices       — scalar int: number of vertices
 *   sources          — int32 or double [E x 1]: edge sources (0-based)
 *   targets          — int32 or double [E x 1]: edge targets (0-based)
 *   k                — scalar int: vertices to remove (0 = original graph)
 *   variant          — scalar int: 0..3
 *   max_iterations   — scalar int: max fitting iterations
 *   epsilon          — scalar double: convergence / bin width
 *   precompute       — scalar int: 0 or 1
 *   keep_multisets   — scalar int: 0 or 1 (optional, default 0)
 *
 * Output:
 *   result — struct with .histogram (double [hist_size x 1]),
 *            .hist_size (int32 scalar), and when keep_multisets is nonzero:
 *            .multisets (double [C(N,k) x E]) and .num_subgraphs (int32).
 */
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

    /* ---- argument count check ---- */
    if (nrhs < 8 || nrhs > 12)
        mexErrMsgIdAndTxt("delta_dress:nrhs",
            "8 to 12 inputs required:\n"
            "  delta_dress_mex(n_vertices, sources, targets, weights, k, "
            "variant, max_iterations, epsilon, precompute, keep_multisets, "
            "offset, stride)\n"
            "  weights may be [] for unweighted.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("delta_dress:nlhs", "At most one output.");

    /* ---- unpack scalars ---- */
    N              = get_scalar_int(prhs[0], "n_vertices");

    /* ---- unpack edge arrays ---- */
    E = (int)mxGetNumberOfElements(prhs[1]);
    U = to_int_array(prhs[1], E, "sources");
    V = to_int_array(prhs[2], E, "targets");

    /* ---- unpack weights (prhs[3], may be empty) ---- */
    if (!mxIsEmpty(prhs[3])) {
        W = (double *)malloc(E * sizeof(double));
        if (!W) { free(U); free(V); mexErrMsgIdAndTxt("delta_dress:oom", "malloc failed"); }
        double *src_w = mxGetPr(prhs[3]);
        for (int i = 0; i < E; i++) W[i] = src_w[i];
    }

    k              = get_scalar_int(prhs[4], "k");
    variant_val    = get_scalar_int(prhs[5], "variant");
    max_iterations = get_scalar_int(prhs[6], "max_iterations");
    epsilon        = get_scalar_double(prhs[7], "epsilon");
    precompute     = (nrhs > 8) ? get_scalar_int(prhs[8], "precompute") : 0;
    keep_ms        = (nrhs > 9) ? get_scalar_int(prhs[9], "keep_multisets") : 0;
    int offset     = (nrhs > 10) ? get_scalar_int(prhs[10], "offset") : 0;
    int stride     = (nrhs > 11) ? get_scalar_int(prhs[11], "stride") : 1;

    /* ---- validate ---- */
    if (variant_val < 0 || variant_val > 3) {
        free(U); free(V); free(W);
        mexErrMsgIdAndTxt("delta_dress:invalidVariant",
                          "variant must be 0..3.");
    }
    if (k < 0) {
        free(U); free(V); free(W);
        mexErrMsgIdAndTxt("delta_dress:invalidK", "k must be >= 0.");
    }

    /* ---- build graph (takes ownership of U, V, W) ---- */
    g = init_dress_graph(N, E, U, V, W,
                         (dress_variant_t)variant_val, precompute);
    if (!g)
        mexErrMsgIdAndTxt("delta_dress:initFailed",
                          "init_dress_graph returned NULL.");

    /* ---- compute delta-k-dress ---- */
    hist = delta_dress_fit_strided(g, k, max_iterations, epsilon, &hist_size,
                    keep_ms,
                    keep_ms ? &ms_ptr : NULL,
                    keep_ms ? &num_sub : NULL,
                    offset, stride);

    /* ---- pack output struct ---- */
    int n_fields = keep_ms ? 4 : 2;
    const char *fields_basic[]  = { "histogram", "hist_size" };
    const char *fields_full[]   = { "histogram", "hist_size", "multisets", "num_subgraphs" };
    plhs[0] = mxCreateStructMatrix(1, 1, n_fields,
                                   keep_ms ? fields_full : fields_basic);

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

    /* multisets and num_subgraphs (when requested) */
    if (keep_ms && ms_ptr) {
        mxArray *m_ms = mxCreateDoubleMatrix((mwSize)num_sub, (mwSize)E, mxREAL);
        {
            /* C stores row-major: ms_ptr[s * E + e]. MATLAB is column-major.
               Transpose during copy: MATLAB(e, s) = ms_ptr[s * E + e]. */
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
        /* Requested but no multisets (e.g. k >= N) */
        mxSetFieldByNumber(plhs[0], 0, 2, mxCreateDoubleMatrix(0, 0, mxREAL));
        mxArray *m_nsub = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
        ((int *)mxGetData(m_nsub))[0] = 0;
        mxSetFieldByNumber(plhs[0], 0, 3, m_nsub);
    }

    /* ---- cleanup ---- */
    free(hist);
    free_dress_graph(g);
}
