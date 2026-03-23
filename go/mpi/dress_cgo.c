/*
 * dress_cgo.c — thin C file that #includes the dress sources so CGo
 * can compile everything (CPU + MPI) in one translation unit.
 *
 * Supports two layouts:
 *   - vendor/src/  (standalone / published module)
 *   - ../../libdress/src/  (monorepo development)
 */
#ifndef DRESS_MPI
#define DRESS_MPI
#endif
#if __has_include("vendor/src/dress.c")
#include "vendor/src/dress.c"
#include "vendor/src/delta_dress_impl.c"
#include "vendor/src/delta_dress.c"
#include "vendor/src/mpi/dress_mpi.c"
#else
#include "../../libdress/src/dress.c"
#include "../../libdress/src/delta_dress_impl.c"
#include "../../libdress/src/delta_dress.c"
#include "../../libdress/src/mpi/dress_mpi.c"
#endif
