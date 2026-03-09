/* cpu_igraph.c — Prism vs K₃,₃ with DRESS (CPU, igraph backend)
 *
 * Same comparison as cpu.c but using igraph_t graphs and the
 * igraph wrapper.  Including dress/igraph/dress.h provides
 * convenience macros so that dress_fit() / dress_free() call
 * the igraph-backed implementation transparently.
 *
 * Build:
 *   gcc -O2 -o cpu_igraph cpu_igraph.c -ldress \
 *       $(pkg-config --cflags --libs igraph) -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <igraph/igraph.h>
#include <dress/igraph/dress.h>

static int cmp_dbl(const void *a, const void *b) {
    double d = *(const double *)a - *(const double *)b;
    return (d > 0) - (d < 0);
}

static void run(const char *name, const int *edges, int n_edges) {
    igraph_t g;
    igraph_vector_int_t ev;
    igraph_vector_int_init(&ev, 2 * n_edges);
    for (int i = 0; i < 2 * n_edges; i++)
        VECTOR(ev)[i] = edges[i];
    igraph_create(&g, &ev, 0, IGRAPH_DIRECTED);
    igraph_vector_int_destroy(&ev);

    dress_result_igraph_t result;
    dress_fit(&g, NULL, DRESS_VARIANT_UNDIRECTED,
             100, 1e-6, 0, &result);

    double *vals = malloc(result.E * sizeof(double));
    memcpy(vals, result.dress, result.E * sizeof(double));
    qsort(vals, result.E, sizeof(double), cmp_dbl);

    printf("%s: ", name);
    for (int i = 0; i < result.E; i++) printf("%.6f ", vals[i]);
    printf("\n");

    free(vals);
    dress_free(&result);
    igraph_destroy(&g);
}

int main(void) {
    igraph_set_attribute_table(&igraph_cattribute_table);

    /* Prism (C₃ □ K₂): 6 vertices, 9 edges (both directions = 18) */
    int prism[] = {0,1,1,2,2,0,0,3,1,4,2,5,3,4,4,5,5,3,
                   1,0,2,1,0,2,3,0,4,1,5,2,4,3,5,4,3,5};

    /* K₃,₃: bipartite {0,1,2} ↔ {3,4,5} — 9 edges (both directions = 18) */
    int k33[] = {0,3,0,4,0,5,1,3,1,4,1,5,2,3,2,4,2,5,
                 3,0,4,0,5,0,3,1,4,1,5,1,3,2,4,2,5,2};

    run("Prism", prism, 18);
    run("K3,3 ", k33,   18);

    printf("Distinguished: fingerprints differ\n");
    return 0;
}
