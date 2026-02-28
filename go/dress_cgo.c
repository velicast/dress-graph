/*
 * dress_cgo.c — thin C file that #includes dress.c and delta_dress.c so CGo
 * can compile everything in one translation unit without needing a separate
 * build step.
 *
 * This file is referenced by the #cgo directive in dress.go.
 */
#include "../libdress/src/dress.c"
#include "../libdress/src/delta_dress.c"
