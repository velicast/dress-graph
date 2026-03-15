/*
 * dress_cgo.c — thin C file that #includes dress.c and delta_dress.c so CGo
 * can compile everything in one translation unit without needing a separate
 * build step.
 *
 * Supports two layouts:
 *   - vendor/src/  (standalone / published module)
 *   - ../libdress/src/  (monorepo development)
 */
#if __has_include("vendor/src/dress.c")
#include "vendor/src/dress.c"
#include "vendor/src/delta_dress_impl.c"
#include "vendor/src/delta_dress.c"
#else
#include "../libdress/src/dress.c"
#include "../libdress/src/delta_dress_impl.c"
#include "../libdress/src/delta_dress.c"
#endif
