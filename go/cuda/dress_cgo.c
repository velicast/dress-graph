/*
 * dress_cgo.c — thin C file that #includes dress.c and delta_dress.c so CGo
 * can compile the CPU init/free functions.  The CUDA fitting functions are
 * expected to be available via -ldress_cuda at link time.
 */
#include "../../libdress/src/dress.c"
#include "../../libdress/src/delta_dress.c"
