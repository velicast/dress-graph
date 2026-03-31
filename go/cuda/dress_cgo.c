/*
 * dress_cgo.c — thin C file that #includes dress.c, delta_dress.c, and the
 * CUDA C wrapper (delta_dress_cuda.c) so CGo can compile everything in one
 * translation unit.  The CUDA kernel object (dress_cuda.o) is linked
 * statically via libdress_cuda_kernel.a.
 *
 * Supports two layouts:
 *   - vendor/src/  (standalone / published module)
 *   - ../../libdress/src/  (monorepo development)
 */
#if __has_include("vendor/src/dress.c")
#include "vendor/src/dress.c"
#include "vendor/src/dress_histogram.c"
#include "vendor/src/delta_dress_impl.c"
#include "vendor/src/delta_dress.c"
#include "vendor/src/nabla_dress_impl.c"
#include "vendor/src/nabla_dress.c"
#include "vendor/src/omp/dress_omp.c"
#include "vendor/src/omp/delta_dress_omp.c"
#include "vendor/src/omp/nabla_dress_omp.c"
#include "vendor/src/cuda/delta_dress_cuda.c"
#include "vendor/src/cuda/nabla_dress_cuda.c"
#else
#include "../../libdress/src/dress.c"
#include "../../libdress/src/dress_histogram.c"
#include "../../libdress/src/delta_dress_impl.c"
#include "../../libdress/src/delta_dress.c"
#include "../../libdress/src/nabla_dress_impl.c"
#include "../../libdress/src/nabla_dress.c"
#include "../../libdress/src/omp/dress_omp.c"
#include "../../libdress/src/omp/delta_dress_omp.c"
#include "../../libdress/src/omp/nabla_dress_omp.c"
#include "../../libdress/src/cuda/delta_dress_cuda.c"
#include "../../libdress/src/cuda/nabla_dress_cuda.c"
#endif
