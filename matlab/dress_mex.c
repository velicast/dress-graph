/*
 * dress_mex.c — MATLAB MEX gateway for the DRESS C library.
 *
 * Compile from the matlab/ directory (MATLAB command window):
 *
 *   mex -O -I../libdress/include CFLAGS="$CFLAGS -fopenmp" LDFLAGS="$LDFLAGS -fopenmp" ...
 *       dress_mex.c ../libdress/src/dress.c -lm
 *
 * Or, if dress.o is already built:
 *
 *   mex -O -I../libdress/include CFLAGS="$CFLAGS -fopenmp" LDFLAGS="$LDFLAGS -fopenmp" ...
 *       dress_mex.c ../dress.o -lm
 *
 * Usage in MATLAB:
 *
 *   result = dress_mex(n_vertices, sources, targets, weights, ...
 *                      variant, max_iterations, epsilon, precompute);
 *
 *   result is a struct with fields:
 *     .sources      — int32 [E x 1]  edge source endpoints (0-based)
 *     .targets      — int32 [E x 1]  edge target endpoints (0-based)
 *     .edge_dress   — double [E x 1] per-edge DRESS similarity
 *     .edge_weight  — double [E x 1] variant-specific edge weight
 *     .node_dress   — double [N x 1] per-node norm
 *     .iterations   — int32 scalar   iterations performed
 *     .delta        — double scalar  final max per-edge change
 */

#include "mex.h"
#if !defined(__OCTAVE__) && !defined(OCTAVE_MEX_FILE)
#include "matrix.h"
#endif
#include "dress/dress.h"

#include <string.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/*  Helper: read a scalar double from a MATLAB argument                */
/* ------------------------------------------------------------------ */
static double get_scalar_double(const mxArray *arg, const char *name)
{
    if (!mxIsDouble(arg) || mxIsComplex(arg) || mxGetNumberOfElements(arg) != 1)
        mexErrMsgIdAndTxt("dress:invalidInput",
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
    /* Also accept double scalars (MATLAB default numeric type) */
    if (mxIsDouble(arg) && !mxIsComplex(arg) && mxGetNumberOfElements(arg) == 1) {
        v = mxGetScalar(arg);
        return (int)v;
    }
    mexErrMsgIdAndTxt("dress:invalidInput",
                      "%s must be a scalar integer (int32 or double).", name);
    return 0; /* unreachable */
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
        mexErrMsgIdAndTxt("dress:invalidInput",
                          "%s must have %d elements.", name, expected_len);

    out = (int *)malloc(expected_len * sizeof(int));
    if (!out)
        mexErrMsgIdAndTxt("dress:malloc", "malloc failed for %s.", name);

    if (mxIsInt32(arg)) {
        memcpy(out, mxGetData(arg), expected_len * sizeof(int));
    } else if (mxIsDouble(arg) && !mxIsComplex(arg)) {
        const double *p = mxGetPr(arg);
        for (i = 0; i < expected_len; i++)
            out[i] = (int)p[i];
    } else {
        free(out);
        mexErrMsgIdAndTxt("dress:invalidInput",
                          "%s must be int32 or double.", name);
    }
    return out;
}

/* ------------------------------------------------------------------ */
/*  Helper: copy a MATLAB double vector to a malloc'd double array     */
/* ------------------------------------------------------------------ */
static double *to_double_array(const mxArray *arg, int expected_len,
                               const char *name)
{
    double *out;
    mwSize n = mxGetNumberOfElements(arg);

    if (!mxIsDouble(arg) || mxIsComplex(arg))
        mexErrMsgIdAndTxt("dress:invalidInput",
                          "%s must be a real double array.", name);
    if ((int)n != expected_len)
        mexErrMsgIdAndTxt("dress:invalidInput",
                          "%s must have %d elements.", name, expected_len);

    out = (double *)malloc(expected_len * sizeof(double));
    if (!out)
        mexErrMsgIdAndTxt("dress:malloc", "malloc failed for %s.", name);
    memcpy(out, mxGetPr(arg), expected_len * sizeof(double));
    return out;
}

/* ------------------------------------------------------------------ */
/*  MEX gateway                                                        */
/* ------------------------------------------------------------------ */

/*
 * MATLAB signature:
 *   result = dress_mex(n_vertices, sources, targets, weights, ...
 *                      variant, max_iterations, epsilon, precompute)
 *
 * Inputs:
 *   n_vertices       — scalar int: number of vertices
 *   sources          — int32 or double [E x 1]: edge sources (0-based)
 *   targets          — int32 or double [E x 1]: edge targets (0-based)
 *   weights          — double [E x 1] or [] for unweighted
 *   variant          — scalar int: 0=UNDIRECTED, 1=DIRECTED,
 *                                  2=FORWARD, 3=BACKWARD
 *   max_iterations   — scalar int: max fitting iterations
 *   epsilon          — scalar double: convergence threshold
 *   precompute       — scalar int: 1 = precompute intercepts, 0 = don't
 *
 * Output:
 *   result — struct with fields listed in the file header.
 */
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

    /* Field names for the output struct */
    const char *field_names[] = {
        "sources", "targets", "edge_dress", "edge_weight",
        "node_dress", "iterations", "delta"
    };

    mxArray *m_src, *m_dst, *m_dress, *m_weight, *m_node, *m_iters, *m_delta;

    /* ---- argument count check ---- */
    if (nrhs != 8)
        mexErrMsgIdAndTxt("dress:nrhs",
            "Eight inputs required:\n"
            "  dress_mex(n_vertices, sources, targets, weights, "
            "variant, max_iterations, epsilon, precompute)");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("dress:nlhs", "At most one output.");

    /* ---- unpack scalar arguments ---- */
    N              = get_scalar_int(prhs[0], "n_vertices");
    variant_val    = get_scalar_int(prhs[4], "variant");
    max_iterations = get_scalar_int(prhs[5], "max_iterations");
    epsilon        = get_scalar_double(prhs[6], "epsilon");
    precompute     = get_scalar_int(prhs[7], "precompute");

    /* ---- unpack edge arrays ---- */
    E = (int)mxGetNumberOfElements(prhs[1]);
    U = to_int_array(prhs[1], E, "sources");
    V = to_int_array(prhs[2], E, "targets");

    /* Optional weights: pass [] in MATLAB for unweighted */
    if (!mxIsEmpty(prhs[3])) {
        W = to_double_array(prhs[3], E, "weights");
    }

    /* ---- validate variant ---- */
    if (variant_val < 0 || variant_val > 3) {
        free(U); free(V); free(W);
        mexErrMsgIdAndTxt("dress:invalidVariant",
                          "variant must be 0 (UNDIRECTED), 1 (DIRECTED), "
                          "2 (FORWARD), or 3 (BACKWARD).");
    }

    /* ---- build graph & fit ---- */
    /* init_dress_graph takes ownership of U, V, W */
    g = init_dress_graph(N, E, U, V, W,
                         (dress_variant_t)variant_val,
                         precompute);
    if (!g)
        mexErrMsgIdAndTxt("dress:initFailed",
                          "init_dress_graph returned NULL.");

    fit(g, max_iterations, epsilon, &iterations, &delta);

    /* ---- pack output struct ---- */
    plhs[0] = mxCreateStructMatrix(1, 1, 7, field_names);

    /* sources — int32 [E x 1] */
    m_src = mxCreateNumericMatrix(E, 1, mxINT32_CLASS, mxREAL);
    memcpy(mxGetData(m_src), g->U, E * sizeof(int));
    mxSetFieldByNumber(plhs[0], 0, 0, m_src);

    /* targets — int32 [E x 1] */
    m_dst = mxCreateNumericMatrix(E, 1, mxINT32_CLASS, mxREAL);
    memcpy(mxGetData(m_dst), g->V, E * sizeof(int));
    mxSetFieldByNumber(plhs[0], 0, 1, m_dst);

    /* edge_dress — double [E x 1] */
    m_dress = mxCreateDoubleMatrix(E, 1, mxREAL);
    memcpy(mxGetPr(m_dress), g->edge_dress, E * sizeof(double));
    mxSetFieldByNumber(plhs[0], 0, 2, m_dress);

    /* edge_weight — double [E x 1] */
    m_weight = mxCreateDoubleMatrix(E, 1, mxREAL);
    memcpy(mxGetPr(m_weight), g->edge_weight, E * sizeof(double));
    mxSetFieldByNumber(plhs[0], 0, 3, m_weight);

    /* node_dress — double [N x 1] */
    m_node = mxCreateDoubleMatrix(N, 1, mxREAL);
    memcpy(mxGetPr(m_node), g->node_dress, N * sizeof(double));
    mxSetFieldByNumber(plhs[0], 0, 4, m_node);

    /* iterations — int32 scalar */
    m_iters = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    ((int *)mxGetData(m_iters))[0] = iterations;
    mxSetFieldByNumber(plhs[0], 0, 5, m_iters);

    /* delta — double scalar */
    m_delta = mxCreateDoubleScalar(delta);
    mxSetFieldByNumber(plhs[0], 0, 6, m_delta);

    /* ---- cleanup ---- */
    free_dress_graph(g);
}
