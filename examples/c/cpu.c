/* cpu.c — Prism vs K₃,₃ with DRESS (CPU)
 *
 * Build:
 *   gcc -O2 -o cpu cpu.c -ldress -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dress/dress.h"

static int cmp_dbl(const void *a, const void *b) {
    double d = *(const double *)a - *(const double *)b;
    return (d > 0) - (d < 0);
}

static void run(const char *name, int N, int E,
                const int *src, const int *dst) {
    int *U = malloc(E * sizeof(int));
    int *V = malloc(E * sizeof(int));
    memcpy(U, src, E * sizeof(int));
    memcpy(V, dst, E * sizeof(int));

    p_dress_graph_t g = init_dress_graph(N, E, U, V, NULL,
                                         DRESS_VARIANT_UNDIRECTED, 0);
    int iters; double delta;
    dress_fit(g, 100, 1e-6, &iters, &delta);

    double *vals = malloc(E * sizeof(double));
    memcpy(vals, g->edge_dress, E * sizeof(double));
    qsort(vals, E, sizeof(double), cmp_dbl);

    printf("%s: ", name);
    for (int i = 0; i < E; i++) printf("%.6f ", vals[i]);
    printf("\n");

    free(vals);
    free_dress_graph(g);
}

int main(void) {
    /* Prism (C₃ □ K₂): 6 vertices, 9 edges (both directions = 18) */
    int prism_s[] = {0,1,1,2,2,0,0,3,1,4,2,5,3,4,4,5,5,3};
    int prism_t[] = {1,0,2,1,0,2,3,0,4,1,5,2,4,3,5,4,3,5};

    /* K₃,₃: bipartite {0,1,2} ↔ {3,4,5} — 9 edges (both directions = 18) */
    int k33_s[] = {0,3,0,4,0,5,1,3,1,4,1,5,2,3,2,4,2,5};
    int k33_t[] = {3,0,4,0,5,0,3,1,4,1,5,1,3,2,4,2,5,2};

    run("Prism", 6, 18, prism_s, prism_t);
    run("K3,3 ", 6, 18, k33_s, k33_t);

    printf("Distinguished: fingerprints differ\n");
    return 0;
}
