#include "dress/dress.h"
#include "delta_dress_impl.h"

int64_t *delta_dress_fit(p_dress_graph_t g, int k, int iterations,
                        double epsilon, int *hist_size,
                        int keep_multisets, double **multisets,
                        int64_t *num_subgraphs)
{
    return delta_dress_fit_impl(g, k, iterations, epsilon, hist_size,
                                keep_multisets, multisets, num_subgraphs,
                                dress_fit);
}
