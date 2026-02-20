#ifndef DRESS_HPP
#define DRESS_HPP

#include "dress/dress.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

// --------------------------------------------------------------------
//  DRESS — RAII C++ wrapper around the C dress_graph_t API.
//
//  Usage:
//    // From raw edge list (copies into malloc'd buffers automatically):
//    DRESS g(N, edges_u, edges_v, weights,
//            DRESS_VARIANT_UNDIRECTED, true);
//
//    // Iterative fitting:
//    auto [iters, delta] = g.fit(100, 1e-6);
//
//    // Access results:
//    double d = g.edgeDress(e);
// --------------------------------------------------------------------
class DRESS {
public:
    // ------- construction from std::vector (copies data) -------

    // Weighted graph.
    DRESS(int N,
               const std::vector<int>& U,
               const std::vector<int>& V,
               const std::vector<double>& W,
               dress_variant_t variant       = DRESS_VARIANT_UNDIRECTED,
               bool precompute_intercepts    = false)
        : g_(nullptr)
    {
        if (U.size() != V.size())
            throw std::invalid_argument("DRESS: U and V must have the same size");
        if (!W.empty() && W.size() != U.size())
            throw std::invalid_argument("DRESS: W must be empty or same size as U/V");

        int E = static_cast<int>(U.size());

        int    *mu = copyToMalloc(U);
        int    *mv = copyToMalloc(V);
        double *mw = W.empty() ? nullptr : copyToMalloc(W);

        g_ = init_dress_graph(N, E, mu, mv, mw, variant,
                              precompute_intercepts ? 1 : 0);
        if (!g_)
            throw std::runtime_error("DRESS: init_dress_graph failed");
    }

    // Unweighted graph.
    DRESS(int N,
               const std::vector<int>& U,
               const std::vector<int>& V,
               dress_variant_t variant       = DRESS_VARIANT_UNDIRECTED,
               bool precompute_intercepts    = false)
        : DRESS(N, U, V, {}, variant, precompute_intercepts)
    {}

    // ------- construction from raw C arrays (takes ownership) -------
    //
    // U, V, W must have been allocated with malloc/calloc.
    // The DRESS (and underlying C code) takes ownership and will
    // free them on destruction.  W may be nullptr for unweighted graphs.
    DRESS(int N, int E,
               int *U, int *V, double *W,
               dress_variant_t variant       = DRESS_VARIANT_UNDIRECTED,
               bool precompute_intercepts    = false)
        : g_(nullptr)
    {
        g_ = init_dress_graph(N, E, U, V, W, variant,
                              precompute_intercepts ? 1 : 0);
        if (!g_)
            throw std::runtime_error("DRESS: init_dress_graph failed");
    }

    // ------- move semantics (no copying — unique ownership) -------

    DRESS(DRESS&& o) noexcept : g_(o.g_) { o.g_ = nullptr; }

    DRESS& operator=(DRESS&& o) noexcept {
        if (this != &o) {
            destroy();
            g_   = o.g_;
            o.g_ = nullptr;
        }
        return *this;
    }

    DRESS(const DRESS&)            = delete;
    DRESS& operator=(const DRESS&) = delete;

    ~DRESS() { destroy(); }

    // ------- fitting -------

    struct FitResult {
        int    iterations;
        double delta;
    };

    // Run iterative dress fitting.
    // Returns the number of iterations performed and the final max delta.
    FitResult fit(int maxIterations, double epsilon) {
        ensureValid();
        int    iters = 0;
        double d     = 0.0;
        ::fit(g_, maxIterations, epsilon, &iters, &d);
        return {iters, d};
    }

    // ------- accessors -------

    int         numVertices()   const { ensureValid(); return g_->N; }
    int         numEdges()      const { ensureValid(); return g_->E; }
    dress_variant_t    variant()    const { ensureValid(); return g_->variant; }

    // Edge endpoints (indexed by edge id 0..E-1).
    int         edgeSource(int e)   const { ensureValid(); return g_->U[e]; }
    int         edgeTarget(int e)   const { ensureValid(); return g_->V[e]; }

    // Edge weight (variant-specific).
    double      edgeWeight(int e)   const { ensureValid(); return g_->edge_weight[e]; }

    // Current dress similarity value for edge e.
    double      edgeDress(int e)    const { ensureValid(); return g_->edge_dress[e]; }

    // Per-node dress norm (sqrt of weighted sum).
    double      nodeDress(int u)    const { ensureValid(); return g_->node_dress[u]; }

    // Pointer-based bulk access (for performance-critical loops).
    const int*    edgeSources()     const { ensureValid(); return g_->U; }
    const int*    edgeTargets()     const { ensureValid(); return g_->V; }
    const double* edgeWeights()     const { ensureValid(); return g_->edge_weight; }
    const double* edgeDressValues() const { ensureValid(); return g_->edge_dress; }
    const double* nodeDressValues() const { ensureValid(); return g_->node_dress; }

    // CSR adjacency access.
    const int*    adjOffset()       const { ensureValid(); return g_->adj_offset; }
    const int*    adjTarget()       const { ensureValid(); return g_->adj_target; }
    const int*    adjEdgeIdx()      const { ensureValid(); return g_->adj_edge_idx; }

    // Access to the underlying C struct (escape hatch for interop).
    p_dress_graph_t       raw()       { return g_; }
    const dress_graph_t*  raw() const { return g_; }

private:
    p_dress_graph_t g_;

    void destroy() noexcept {
        if (g_) { free_dress_graph(g_); g_ = nullptr; }
    }

    void ensureValid() const {
        if (!g_)
            throw std::logic_error("DRESS: accessing a moved-from or null graph");
    }

    // Allocate a malloc'd copy of a std::vector for handoff to the C API.
    template <typename T>
    static T* copyToMalloc(const std::vector<T>& v) {
        T* p = static_cast<T*>(std::malloc(v.size() * sizeof(T)));
        if (!p) throw std::bad_alloc();
        std::memcpy(p, v.data(), v.size() * sizeof(T));
        return p;
    }
};

#endif // DRESS_HPP
