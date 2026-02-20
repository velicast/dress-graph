#ifndef DRESS_H
#define DRESS_H

#ifdef __cplusplus
extern "C" {
#endif

// dress graph variant: determines how adjacency lists are constructed
// from the input edge list.
//
//   UNDIRECTED: N[u] = N(u)           (symmetric neighborhood)
//   DIRECTED:   N[u] = in(u) + out(u) (union of in- and out-neighbors)
//   FORWARD:    N[u] = out(u)         (only outgoing neighbors)
//   BACKWARD:   N[u] = in(u)          (only incoming neighbors)
typedef enum __dress_variant_t {
    DRESS_VARIANT_UNDIRECTED = 0,
    DRESS_VARIANT_DIRECTED   = 1,
    DRESS_VARIANT_FORWARD    = 2,
    DRESS_VARIANT_BACKWARD   = 3
} dress_variant_t;



// dress graph structure.
//
// Stores the augmented variant adjacency in CSR (Compressed Sparse Row)
// format, with per-edge weight and similarity arrays. An optional set of
// precomputed neighborhood intercepts accelerates iterative fitting from
// O(deg_u + deg_v) to O(|N[u] ∩ N[v]|) per edge.
//
// Memory layout (S = total half-edges, T = total intercept entries):
//   Permanent:  12N + 52E + 8T  bytes  (with intercepts)
//               12N + 48E       bytes  (without intercepts)
typedef struct __dress_graph_t {
    dress_variant_t variant;       // graph variant (determines adjacency)
    int      N;                    // number of vertices
    int      E;                    // number of input edges

    int     *U;                    // [E]   input edge sources (owned)
    int     *V;                    // [E]   input edge targets (owned)

    // CSR variant adjacency — sorted per-node neighbor lists stored flat.
    // Neighbors of node u are adj_target[adj_offset[u] .. adj_offset[u+1]).
    int     *adj_offset;           // [N+1] CSR row offsets
    int     *adj_target;           // [S]   neighbor vertex ids
    int     *adj_edge_idx;         // [S]   maps half-edge to input edge index

    // Per-edge arrays — indexed by edge id 0..E-1.
    double  *edge_weight;          // [E]   variant-specific edge weight
    double  *edge_dress;           // [E]   current dress values
    double  *edge_dress_next;      // [E]   next-iteration dress values (double-buffer)

    // Per-node arrays — indexed by vertex id 0..N-1.
    double  *node_dress;           // [N]   sqrt of weighted dress sum for node

    // Precomputed intercepts (allocated only when precompute_intercepts == 1).
    // For edge e = (u,v), common neighbors are stored at
    // intercept_edge_ux/vx[intercept_offset[e] .. intercept_offset[e+1]).
    int      precompute_intercepts;
    int     *intercept_offset;     // [E+1] CSR-style offsets into intercept arrays
    int     *intercept_edge_ux;    // [T]   edge index for (u,x) per common neighbor x
    int     *intercept_edge_vx;    // [T]   edge index for (v,x) per common neighbor x
} dress_graph_t, *p_dress_graph_t;

// Construct a dress graph from an edge list.
// Takes ownership of U, V, W (freed by free_dress_graph). W may be NULL (unweighted).
p_dress_graph_t init_dress_graph(int N, int E, int *U, int *V,
                                 double *W, dress_variant_t variant,
                                 int precompute_intercepts);

// Run iterative dress fitting for at most max_iterations, stopping early
// when the maximum per-edge change falls below epsilon.
// On return, *iterations and *delta (if non-NULL) hold the iteration
// count and final max delta.
void   fit(p_dress_graph_t g, int max_iterations, double epsilon,
           int *iterations, double *delta);

// Free all memory associated with the dress graph (including U, V).
void   free_dress_graph(p_dress_graph_t g);

#ifdef __cplusplus
}
#endif

#endif // DRESS_H