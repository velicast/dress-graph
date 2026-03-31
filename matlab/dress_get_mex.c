/*
 * dress_get_mex.c — MEX gateway: query DRESS value on a fitted graph.
 *
 * Usage:
 *   val = dress_get_mex(ptr, u, v, max_iterations, epsilon, edge_weight)
 *
 * ptr is the uint64 handle from dress_init_mex.
 * Returns a scalar double.
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
    p_dress_graph_t g;
    int u, v, max_iterations;
    double epsilon, edge_weight, val;

    if (nrhs != 6)
        mexErrMsgIdAndTxt("dress:nrhs",
            "Six inputs required: dress_get_mex(ptr, u, v, max_iterations, epsilon, edge_weight)");

    g = (p_dress_graph_t)(uintptr_t)(*((uint64_t *)mxGetData(prhs[0])));
    if (!g)
        mexErrMsgIdAndTxt("dress:nullPtr", "Graph pointer is NULL (already closed?).");

    u              = (int)mxGetScalar(prhs[1]);
    v              = (int)mxGetScalar(prhs[2]);
    max_iterations = (int)mxGetScalar(prhs[3]);
    epsilon        = mxGetScalar(prhs[4]);
    edge_weight    = mxGetScalar(prhs[5]);

    val = dress_get(g, u, v, max_iterations, epsilon, edge_weight);

    plhs[0] = mxCreateDoubleScalar(val);
}
