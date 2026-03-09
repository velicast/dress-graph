/*
 * dress_cgo.c — thin C file that #includes the dress sources so CGo
 * can compile everything (CPU + MPI) in one translation unit.
 */
#include "../../libdress/src/dress.c"
#include "../../libdress/src/delta_dress_impl.c"
#include "../../libdress/src/delta_dress.c"
#include "../../libdress/src/mpi/dress_mpi.c"
