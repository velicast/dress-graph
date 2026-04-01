/*
 * dress_result_mex.c — MEX gateway: extract current results from a
 *                      persistent DRESS graph without freeing it.
 *
 * Usage:
 *   result = dress_result_mex(ptr)
 *
 * ptr is the uint64 handle from dress_init_mex.
 * Returns a struct with fields: sources, targets, edge_dress,
 *                               edge_weight, vertex_dress.
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

#include <string.h>
#include <stdint.h>

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    p_dress_graph_t g;
    const char *field_names[] = {
        "sources", "targets", "edge_dress", "edge_weight", "vertex_dress"
    };
    mxArray *m_src, *m_tgt, *m_dress, *m_weight, *m_node;

    if (nrhs != 1)
        mexErrMsgIdAndTxt("dress:nrhs", "One input required: dress_result_mex(ptr)");

    g = (p_dress_graph_t)(uintptr_t)(*((uint64_t *)mxGetData(prhs[0])));
    if (!g)
        mexErrMsgIdAndTxt("dress:nullPtr", "Graph pointer is NULL (already closed?).");

    plhs[0] = mxCreateStructMatrix(1, 1, 5, field_names);

    m_src    = mxCreateNumericMatrix(g->E, 1, mxINT32_CLASS, mxREAL);
    m_tgt    = mxCreateNumericMatrix(g->E, 1, mxINT32_CLASS, mxREAL);
    m_dress  = mxCreateDoubleMatrix(g->E, 1, mxREAL);
    m_weight = mxCreateDoubleMatrix(g->E, 1, mxREAL);
    m_node   = mxCreateDoubleMatrix(g->N, 1, mxREAL);

    memcpy(mxGetData(m_src),    g->U,          g->E * sizeof(int));
    memcpy(mxGetData(m_tgt),    g->V,          g->E * sizeof(int));
    memcpy(mxGetPr(m_dress),    g->edge_dress, g->E * sizeof(double));
    memcpy(mxGetPr(m_weight),   g->edge_weight,g->E * sizeof(double));
    memcpy(mxGetPr(m_node),     g->vertex_dress, g->N * sizeof(double));

    mxSetFieldByNumber(plhs[0], 0, 0, m_src);
    mxSetFieldByNumber(plhs[0], 0, 1, m_tgt);
    mxSetFieldByNumber(plhs[0], 0, 2, m_dress);
    mxSetFieldByNumber(plhs[0], 0, 3, m_weight);
    mxSetFieldByNumber(plhs[0], 0, 4, m_node);
}
