/* mpi_igraph.c — Rook vs Shrikhande with Δ¹-DRESS (MPI, igraph backend)
 *
 * Same comparison as mpi.c but using igraph_t graphs and the
 * MPI-distributed igraph wrapper.  Including dress/mpi/igraph/dress.h
 * redirects delta_dress_fit() to the MPI implementation.
 *
 * Build & run:
 *   mpicc -O2 -o mpi_igraph mpi_igraph.c -ldress \
 *       $(pkg-config --cflags --libs igraph) -lm
 *   mpirun -np 4 ./mpi_igraph
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <igraph/igraph.h>
#include <dress/mpi/igraph/dress.h>

static igraph_t make_graph(const int *edges, int n_edges) {
    igraph_t g;
    igraph_vector_int_t ev;
    igraph_vector_int_init(&ev, 2 * n_edges);
    for (int i = 0; i < 2 * n_edges; i++)
        VECTOR(ev)[i] = edges[i];
    igraph_create(&g, &ev, 0, IGRAPH_DIRECTED);
    igraph_vector_int_destroy(&ev);
    return g;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    igraph_set_attribute_table(&igraph_cattribute_table);

    /* Rook L₂(4) = K₄ □ K₄ — 16 vertices, 96 directed edges */
    int rook_e[] = {0,1,0,4,0,2,0,8,0,3,0,12,1,5,1,2,1,9,1,3,1,13,2,6,2,10,2,3,2,14,3,7,3,11,3,15,4,5,4,6,4,8,4,7,4,12,5,6,5,9,5,7,5,13,6,10,6,7,6,14,7,11,7,15,8,9,8,10,8,11,8,12,9,10,9,11,9,13,10,11,10,14,11,15,12,13,12,14,12,15,13,14,13,15,14,15,
                    1,0,4,0,2,0,8,0,3,0,12,0,5,1,2,1,9,1,3,1,13,1,6,2,10,2,3,2,14,2,7,3,11,3,15,3,5,4,6,4,8,4,7,4,12,4,6,5,9,5,7,5,13,5,10,6,7,6,14,6,11,7,15,7,9,8,10,8,11,8,12,8,10,9,11,9,13,9,11,10,14,10,15,11,13,12,14,12,15,12,14,13,15,13,15,14};

    /* Shrikhande — 16 vertices, 96 directed edges */
    int shri_e[] = {0,4,0,12,0,1,0,3,0,5,0,15,1,5,1,13,1,2,1,6,1,12,2,6,2,14,2,3,2,7,2,13,3,7,3,15,3,4,3,14,4,8,4,5,4,7,4,9,5,9,5,6,5,10,6,10,6,7,6,11,7,11,7,8,8,12,8,9,8,11,8,13,9,13,9,10,9,14,10,14,10,11,10,15,11,15,11,12,12,13,12,15,13,14,14,15,
                    4,0,12,0,1,0,3,0,5,0,15,0,5,1,13,1,2,1,6,1,12,1,6,2,14,2,3,2,7,2,13,2,7,3,15,3,4,3,14,3,8,4,5,4,7,4,9,4,9,5,6,5,10,5,10,6,7,6,11,6,11,7,8,7,12,8,9,8,11,8,13,8,13,9,10,9,14,9,14,10,11,10,15,10,15,11,12,11,13,12,15,12,14,13,15,14};

    igraph_t rook = make_graph(rook_e, 96);
    igraph_t shri = make_graph(shri_e, 96);

    delta_dress_result_igraph_t dr, ds;
    delta_dress_fit(&rook, NULL, DRESS_VARIANT_UNDIRECTED,
                    1, 100, 1e-6, 0, &dr);
    delta_dress_fit(&shri, NULL, DRESS_VARIANT_UNDIRECTED,
                    1, 100, 1e-6, 0, &ds);

    if (rank == 0) {
        printf("Rook:       %d bins\n", dr.hist_size);
        printf("Shrikhande: %d bins\n", ds.hist_size);

        int same = (dr.hist_size == ds.hist_size) &&
                   memcmp(dr.histogram, ds.histogram,
                          (size_t)dr.hist_size * sizeof(int64_t)) == 0;
        printf("Histograms differ: %s\n", same ? "no" : "yes");
    }

    delta_dress_free(&dr);
    delta_dress_free(&ds);
    igraph_destroy(&rook);
    igraph_destroy(&shri);

    MPI_Finalize();
    return 0;
}
