/**
 * dress/cuda/dress.h — GPU-accelerated DRESS via include-based switching.
 *
 * Drop-in replacement for dress/dress.h + dress/delta_dress.h.
 * Including this header redirects dress_fit() and delta_dress_fit()
 * to their CUDA implementations — no source changes required:
 *
 *   // CPU
 *   #include "dress/dress.h"
 *   dress_fit(g, 100, 1e-6, &iters, &delta);
 *
 *   // CUDA — same call, different include
 *   #include "dress/cuda/dress.h"
 *   dress_fit(g, 100, 1e-6, &iters, &delta);
 *
 * Do not include both this header and dress/dress.h in the same
 * translation unit — the macros will conflict.
 */

#ifndef DRESS_CUDA_REDIRECT_H
#define DRESS_CUDA_REDIRECT_H

#include "dress/dress.h"
#include "dress/cuda/dress_cuda.h"

/* Redirect CPU symbols to CUDA implementations. */
#define dress_fit       dress_fit_cuda
#define delta_dress_fit delta_dress_fit_cuda
#define delta_dress_fit_strided delta_dress_fit_cuda_strided

#endif /* DRESS_CUDA_REDIRECT_H */
