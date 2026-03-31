/* mpi_omp.c — Rook vs Shrikhande with Δ¹-DRESS (MPI + OpenMP)
 *
 * Keeps multisets and compares them to guarantee distinguishability.
 *
 * Build & run:
 *   mpicc -O2 -fopenmp -o mpi_omp mpi_omp.c -ldress -lm
 *   mpirun -np 4 ./mpi_omp
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "dress/mpi/omp/dress.h"

typedef struct {
    dress_hist_pair_t *hist;
    int      hs;
    double  *multisets;
    int64_t  ns;
} result_t;

static int hist_equal(const dress_hist_pair_t *a, const dress_hist_pair_t *b,
                      int hs_a, int hs_b) {
    if (hs_a != hs_b) return 0;
    for (int i = 0; i < hs_a; i++) {
        if (a[i].value != b[i].value || a[i].count != b[i].count) return 0;
    }
    return 1;
}

static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    if (isnan(x) && isnan(y)) return 0;
    if (isnan(x)) return 1;
    if (isnan(y)) return -1;
    return (x > y) - (x < y);
}

static result_t do_fit(int N, int E, const int *src, const int *dst) {
    int *U = malloc(E * sizeof(int));
    int *V = malloc(E * sizeof(int));
    memcpy(U, src, E * sizeof(int));
    memcpy(V, dst, E * sizeof(int));

    p_dress_graph_t g = dress_init_graph(N, E, U, V, NULL, NULL,
        DRESS_VARIANT_UNDIRECTED, 0);
    result_t r;
    r.multisets = NULL;
    r.hist = dress_delta_fit(g, /*k=*/1, 100, 1e-6, 0, 0, &r.hs, 1, &r.multisets, &r.ns);
    dress_free_graph(g);
    return r;
}

/* Sort each row, then sort rows lexicographically (canonical form) */
static void sort_multisets(double *m, int64_t ns, int E) {
    for (int64_t i = 0; i < ns; i++)
        qsort(m + i * E, E, sizeof(double), cmp_dbl);
    double *tmp = malloc(E * sizeof(double));
    for (int64_t i = 0; i < ns - 1; i++)
        for (int64_t j = i + 1; j < ns; j++) {
            for (int c = 0; c < E; c++) {
                int r = cmp_dbl(m + i * E + c, m + j * E + c);
                if (r > 0) {
                    memcpy(tmp,       m + i * E, E * sizeof(double));
                    memcpy(m + i * E, m + j * E, E * sizeof(double));
                    memcpy(m + j * E, tmp,       E * sizeof(double));
                    break;
                }
                if (r < 0) break;
            }
        }
    free(tmp);
}

static int multisets_equal(const double *a, const double *b,
                           int64_t ns_a, int64_t ns_b, int E) {
    if (ns_a != ns_b) return 0;
    for (int64_t i = 0; i < ns_a * E; i++) {
        if (isnan(a[i]) && isnan(b[i])) continue;
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rook_s[] = {0,1,0,4,0,2,0,8,0,3,0,12,1,5,1,2,1,9,1,3,1,13,2,6,2,10,2,3,2,14,3,7,3,11,3,15,4,5,4,6,4,8,4,7,4,12,5,6,5,9,5,7,5,13,6,10,6,7,6,14,7,11,7,15,8,9,8,10,8,11,8,12,9,10,9,11,9,13,10,11,10,14,11,15,12,13,12,14,12,15,13,14,13,15,14,15};
    int rook_t[] = {1,0,4,0,2,0,8,0,3,0,12,0,5,1,2,1,9,1,3,1,13,1,6,2,10,2,3,2,14,2,7,3,11,3,15,3,5,4,6,4,8,4,7,4,12,4,6,5,9,5,7,5,13,5,10,6,7,6,14,6,11,7,15,7,9,8,10,8,11,8,12,8,10,9,11,9,13,9,11,10,14,10,15,11,13,12,14,12,15,12,14,13,15,13,15,14};

    int shri_s[] = {0,4,0,12,0,1,0,3,0,5,0,15,1,5,1,13,1,2,1,6,1,12,2,6,2,14,2,3,2,7,2,13,3,7,3,15,3,4,3,14,4,8,4,5,4,7,4,9,5,9,5,6,5,10,6,10,6,7,6,11,7,11,7,8,8,12,8,9,8,11,8,13,9,13,9,10,9,14,10,14,10,11,10,15,11,15,11,12,12,13,12,15,13,14,14,15};
    int shri_t[] = {4,0,12,0,1,0,3,0,5,0,15,0,5,1,13,1,2,1,6,1,12,1,6,2,14,2,3,2,7,2,13,2,7,3,15,3,4,3,14,3,8,4,5,4,7,4,9,4,9,5,6,5,10,5,10,6,7,6,11,6,11,7,8,7,12,8,9,8,11,8,13,8,13,9,10,9,14,9,14,10,11,10,15,10,15,11,12,11,13,12,15,12,14,13,15,14};

    const int N = 16, E = 96;
    result_t dr = do_fit(N, E, rook_s, rook_t);
    result_t ds = do_fit(N, E, shri_s, shri_t);

    if (rank == 0) {
        printf("Rook:       %d bins, %ld subgraphs\n", dr.hs, (long)dr.ns);
        printf("Shrikhande: %d bins, %ld subgraphs\n", ds.hs, (long)ds.ns);

        int hist_same = hist_equal(dr.hist, ds.hist, dr.hs, ds.hs);
        printf("Histograms differ:  %s\n", hist_same ? "no" : "yes");

        sort_multisets(dr.multisets, dr.ns, E);
        sort_multisets(ds.multisets, ds.ns, E);
        int ms_same = multisets_equal(dr.multisets, ds.multisets,
                                      dr.ns, ds.ns, E);
        printf("Multisets differ:   %s\n", ms_same ? "no" : "yes");
    }

    free(dr.hist); free(dr.multisets);
    free(ds.hist); free(ds.multisets);

    MPI_Finalize();
    return 0;
}
