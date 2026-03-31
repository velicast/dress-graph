/*
 * dress_cgo.c — thin C file that #includes the dress sources so CGo
 * can compile everything (CPU + OMP + MPI) in one translation unit.
 */
#ifndef DRESS_MPI
#define DRESS_MPI
#endif
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
#include "vendor/src/mpi/dress_mpi.c"
#else
#include "../../../libdress/src/dress.c"
#include "../../../libdress/src/dress_histogram.c"
#include "../../../libdress/src/delta_dress_impl.c"
#include "../../../libdress/src/delta_dress.c"
#include "../../../libdress/src/nabla_dress_impl.c"
#include "../../../libdress/src/nabla_dress.c"
#include "../../../libdress/src/omp/dress_omp.c"
#include "../../../libdress/src/omp/delta_dress_omp.c"
#include "../../../libdress/src/omp/nabla_dress_omp.c"
#include "../../../libdress/src/mpi/dress_mpi.c"
#endif
