/*
 * dress_cgo.c — thin C file that #includes all needed C sources so CGo
 * can compile everything in one translation unit.  The CUDA kernel object
 * (dress_cuda.o) is linked statically via libdress_cuda.a.
 *
 * Supports two layouts:
 *   - vendor/src/  (standalone / published module)
 *   - ../../../libdress/src/  (monorepo development)
 */
#ifndef DRESS_MPI
#define DRESS_MPI
#endif
#ifndef DRESS_CUDA
#define DRESS_CUDA
#endif
#if __has_include("vendor/src/dress.c")
#include "vendor/src/dress.c"
#include "vendor/src/delta_dress_impl.c"
#include "vendor/src/delta_dress.c"
#include "vendor/src/cuda/delta_dress_cuda.c"
#include "vendor/src/mpi/dress_mpi.c"
#else
#include "../../../libdress/src/dress.c"
#include "../../../libdress/src/delta_dress_impl.c"
#include "../../../libdress/src/delta_dress.c"
#include "../../../libdress/src/cuda/delta_dress_cuda.c"
#include "../../../libdress/src/mpi/dress_mpi.c"
#endif
