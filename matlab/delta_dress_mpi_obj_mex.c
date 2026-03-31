/*
 * delta_dress_mpi_obj_mex.c — MEX gateway: MPI delta-fit on a persistent DRESS graph.
 *
 * Usage:
 *   result = delta_dress_mpi_obj_mex(ptr, k, max_iterations, epsilon, keep_multisets)
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
#include "dress/mpi/dress_mpi.h"

#include <stdint.h>

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
    p_dress_graph_t g;
    int k;
    int max_iterations;
    double epsilon;
    int keep_ms;
    int n_samples = 0;
    unsigned int seed = 0;
    int compute_hist = 1;
    int hist_size = 0;
    dress_hist_pair_t *hist;
    double *ms_ptr = NULL;
    int64_t num_sub = 0;

    if (nrhs != 5)
        mexErrMsgIdAndTxt("dress_mpi:nrhs",
            "Five inputs required: delta_dress_mpi_obj_mex(ptr, k, max_iterations, epsilon, keep_multisets)");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("dress_mpi:nlhs", "At most one output.");

    g = (p_dress_graph_t)(uintptr_t)(*((uint64_t *)mxGetData(prhs[0])));
    if (!g)
        mexErrMsgIdAndTxt("dress_mpi:nullPtr", "Graph pointer is NULL (already closed?).");

    k              = (int)mxGetScalar(prhs[1]);
    max_iterations = (int)mxGetScalar(prhs[2]);
    epsilon        = mxGetScalar(prhs[3]);
    keep_ms        = (int)mxGetScalar(prhs[4]);

    if (k < 0)
        mexErrMsgIdAndTxt("dress_mpi:invalidK", "k must be >= 0.");

    dress_mpi_init();
    hist = dress_delta_fit_mpi_world(g, k, max_iterations, epsilon,
                                     n_samples, seed,
                                     compute_hist ? &hist_size : NULL,
                                     keep_ms,
                                     keep_ms ? &ms_ptr : NULL,
                                     &num_sub);

    {
        int n_fields = keep_ms ? 3 : 1;
        const char *fields_basic[] = { "histogram" };
        const char *fields_full[] = { "histogram", "multisets", "num_subgraphs" };
        plhs[0] = mxCreateStructMatrix(1, 1, n_fields,
                                       keep_ms ? fields_full : fields_basic);
    }

    mxSetFieldByNumber(plhs[0], 0, 0, build_histogram_struct(hist, hist_size));

    if (keep_ms && ms_ptr) {
        int E = g->E;
        mxArray *m_ms = mxCreateDoubleMatrix((mwSize)num_sub, (mwSize)E, mxREAL);
        double *dst = mxGetPr(m_ms);
        int64_t s;
        int e;

        for (s = 0; s < num_sub; s++)
            for (e = 0; e < E; e++)
                dst[e * (mwSize)num_sub + s] = ms_ptr[s * E + e];

        mxSetFieldByNumber(plhs[0], 0, 1, m_ms);

        {
            mxArray *m_nsub = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
            ((int *)mxGetData(m_nsub))[0] = (int)num_sub;
            mxSetFieldByNumber(plhs[0], 0, 2, m_nsub);
        }

        free(ms_ptr);
    } else if (keep_ms) {
        mxSetFieldByNumber(plhs[0], 0, 1, mxCreateDoubleMatrix(0, 0, mxREAL));
        {
            mxArray *m_nsub = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
            ((int *)mxGetData(m_nsub))[0] = 0;
            mxSetFieldByNumber(plhs[0], 0, 2, m_nsub);
        }
    }

    free(hist);
}