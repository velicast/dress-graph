/*
 * dress_free_mex.c — MEX gateway: free a persistent DRESS graph.
 *
 * Usage:
 *   dress_free_mex(ptr)
 *
 * ptr is the uint64 handle from dress_init_mex.
 */

#include "mex.h"
#if !defined(__OCTAVE__) && !defined(OCTAVE_MEX_FILE)
#include "matrix.h"
#endif
#include "dress/dress.h"

#include <stdint.h>

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    p_dress_graph_t g;

    if (nrhs != 1)
        mexErrMsgIdAndTxt("dress:nrhs", "One input required: dress_free_mex(ptr)");

    g = (p_dress_graph_t)(uintptr_t)(*((uint64_t *)mxGetData(prhs[0])));
    if (g) {
        free_dress_graph(g);
    }
}
