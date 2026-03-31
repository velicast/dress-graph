/*
 * dress_fit_obj_mex.c — MEX gateway: fit a persistent DRESS graph.
 *
 * Usage:
 *   result = dress_fit_obj_mex(ptr, max_iterations, epsilon)
 *
 * ptr is the uint64 handle from dress_init_mex.
 * Returns a struct with fields: iterations, delta.
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

#include <stdint.h>

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    int max_iterations;
    double epsilon;
    p_dress_graph_t g;
    int iterations = 0;
    double delta = 0.0;
    const char *field_names[] = {"iterations", "delta"};

    if (nrhs != 3)
        mexErrMsgIdAndTxt("dress:nrhs",
            "Three inputs required: dress_fit_obj_mex(ptr, max_iterations, epsilon)");

    g = (p_dress_graph_t)(uintptr_t)(*((uint64_t *)mxGetData(prhs[0])));
    if (!g)
        mexErrMsgIdAndTxt("dress:nullPtr", "Graph pointer is NULL (already closed?).");

    max_iterations = (int)mxGetScalar(prhs[1]);
    epsilon        = mxGetScalar(prhs[2]);

    dress_fit(g, max_iterations, epsilon, &iterations, &delta);

    plhs[0] = mxCreateStructMatrix(1, 1, 2, field_names);
    mxSetFieldByNumber(plhs[0], 0, 0, mxCreateDoubleScalar((double)iterations));
    mxSetFieldByNumber(plhs[0], 0, 1, mxCreateDoubleScalar(delta));
}
