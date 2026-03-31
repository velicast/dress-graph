#ifndef DRESS_HPP
#define DRESS_HPP

#include "dress/dress.h"

#include <cstdlib>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <vector>

namespace dress {

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
             const std::vector<double>& NW = {},
               dress_variant_t variant       = DRESS_VARIANT_UNDIRECTED,
               bool precompute_intercepts    = false)
        : g_(nullptr)
    {
        if (U.size() != V.size())
            throw std::invalid_argument("DRESS: U and V must have the same size");
        if (!W.empty() && W.size() != U.size())
            throw std::invalid_argument("DRESS: W must be empty or same size as U/V");
        if (!NW.empty() && static_cast<int>(NW.size()) != N)
            throw std::invalid_argument("DRESS: NW must be empty or size N");

        int E = static_cast<int>(U.size());

        int    *mu  = copyToMalloc(U);
        int    *mv  = copyToMalloc(V);
        double *mw  = W.empty() ? nullptr : copyToMalloc(W);
        double *mnw = NW.empty() ? nullptr : copyToMalloc(NW);

        g_ = dress_init_graph(N, E, mu, mv, mw, mnw, variant,
                              precompute_intercepts ? 1 : 0);
        if (!g_)
            throw std::runtime_error("DRESS: dress_init_graph failed");
    }

    // Unweighted graph (no edge weights, no node weights).
    DRESS(int N,
               const std::vector<int>& U,
               const std::vector<int>& V,
               dress_variant_t variant       = DRESS_VARIANT_UNDIRECTED,
               bool precompute_intercepts    = false)
        : DRESS(N, U, V, {}, {}, variant, precompute_intercepts)
    {}

    // ------- construction from raw C arrays (takes ownership) -------
    //
    // U, V, W, NW must have been allocated with malloc/calloc.
    // The DRESS (and underlying C code) takes ownership and will
    // free them on destruction.  W may be nullptr for unweighted graphs.
    // NW may be nullptr (all node weights 1.0).
    DRESS(int N, int E,
               int *U, int *V, double *W, double *NW = nullptr,
               dress_variant_t variant       = DRESS_VARIANT_UNDIRECTED,
               bool precompute_intercepts    = false)
        : g_(nullptr)
    {
        g_ = dress_init_graph(N, E, U, V, W, NW, variant,
                              precompute_intercepts ? 1 : 0);
        if (!g_)
            throw std::runtime_error("DRESS: dress_init_graph failed");
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
        ::dress_fit(g_, maxIterations, epsilon, &iters, &d);
        return {iters, d};
    }

    // ------- delta fitting (Δ^k-DRESS) -------

    struct DeltaFitResult {
        std::vector<std::pair<double, int64_t>> histogram;   // sorted (edge_dress_value, count) pairs
        std::vector<double>       multisets;   // C(N,k) * E row-major, NaN = removed
        int64_t                   num_subgraphs; // C(N,k) — number of rows
    };

    // Run Δ^k-DRESS: enumerate all C(N,k) node-deletion subsets,
    // fit DRESS on each, and accumulate edge values into a histogram.
    //
    // Parameters:
    //   k              – deletion depth (0 = original graph)
    //   maxIterations  – max DRESS iterations per subgraph
    //   epsilon        – convergence tolerance
    //   keepMultisets  – if true, also return per-subgraph edge values
    //   computeHistogram – if false, skip histogram construction
    //
    // Returns a DeltaFitResult with the histogram (and optionally multisets).
    DeltaFitResult deltaFit(int k, int maxIterations, double epsilon,
                            int nSamples = 0, unsigned int seed = 0,
                            bool keepMultisets = false,
                            bool computeHistogram = true) {
        ensureValid();
        int hsize = 0;
        int E = g_->E;

        double *ms_ptr = nullptr;
        int64_t cnk = 0;
        dress_hist_pair_t *h = ::dress_delta_fit_strided(g_, k, maxIterations, epsilon,
                                                     nSamples, seed,
                                                     computeHistogram ? &hsize : nullptr,
                                                     keepMultisets ? 1 : 0,
                                                     keepMultisets ? &ms_ptr : nullptr,
                                                     &cnk,
                                                     0, 1);
        DeltaFitResult result;
        if (h && hsize > 0) {
            for (int i = 0; i < hsize; i++) {
                result.histogram.emplace_back(h[i].value, h[i].count);
            }
            std::free(h);
        }
        result.num_subgraphs = cnk;
        if (keepMultisets && ms_ptr) {
            result.multisets.assign(ms_ptr, ms_ptr + cnk * E);
            std::free(ms_ptr);
        }
        return result;
    }

    // ------- nabla fitting (∇^k-DRESS) -------

    struct NablaFitResult {
        std::vector<std::pair<double, int64_t>> histogram;   // sorted (edge_dress_value, count) pairs
        std::vector<double>       multisets;   // P(N,k) * E row-major
        int64_t                   num_tuples;  // P(N,k) — number of rows
    };

    // Run ∇^k-DRESS: enumerate all P(N,k) ordered k-tuples,
    // mark each tuple with generic injective node weights,
    // fit DRESS on each marked graph, and accumulate edge values
    // into a histogram.
    NablaFitResult nablaFit(int k, int maxIterations, double epsilon,
                            int nSamples = 0, unsigned int seed = 0,
                            bool keepMultisets = false,
                            bool computeHistogram = true) {
        ensureValid();
        int hsize = 0;
        int E = g_->E;

        double *ms_ptr = nullptr;
        int64_t pnk = 0;
        dress_hist_pair_t *h = ::dress_nabla_fit(g_, k, maxIterations, epsilon,
                                                 nSamples, seed,
                                                 computeHistogram ? &hsize : nullptr,
                                                 keepMultisets ? 1 : 0,
                                                 keepMultisets ? &ms_ptr : nullptr,
                                                 &pnk);
        NablaFitResult result;
        if (h && hsize > 0) {
            for (int i = 0; i < hsize; i++) {
                result.histogram.emplace_back(h[i].value, h[i].count);
            }
            std::free(h);
        }
        result.num_tuples = pnk;
        if (keepMultisets && ms_ptr) {
            result.multisets.assign(ms_ptr, ms_ptr + pnk * E);
            std::free(ms_ptr);
        }
        return result;
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

    // ------- virtual-edge query -------

    // Query the DRESS value for any vertex pair (u, v).
    //
    // If edge (u,v) exists, returns its converged edge_dress value.
    // If the edge does not exist (virtual edge), estimates the value
    // using a local fixed-point iteration against the frozen steady state.
    // Useful for link prediction.
    //
    // The graph must have been fitted (fit()) before calling this.
    // edgeWeight is the hypothetical weight of the virtual edge (ignored
    // when the edge already exists).  Pass 1.0 for unweighted graphs.
    double get(int u, int v,
               int maxIterations = 100,
               double epsilon = 1e-6,
               double edgeWeight = 1.0) const {
        ensureValid();
        return ::dress_get(g_, u, v, maxIterations, epsilon, edgeWeight);
    }

    // CSR adjacency access.
    const int*    adjOffset()       const { ensureValid(); return g_->adj_offset; }
    const int*    adjTarget()       const { ensureValid(); return g_->adj_target; }
    const int*    adjEdgeIdx()      const { ensureValid(); return g_->adj_edge_idx; }

    // Access to the underlying C struct (escape hatch for interop).
    p_dress_graph_t       raw()       { return g_; }
    const dress_graph_t*  raw() const { return g_; }

protected:
    void ensureValid() const {
        if (!g_)
            throw std::logic_error("DRESS: accessing a moved-from or null graph");
    }

private:
    p_dress_graph_t g_;

    void destroy() noexcept {
        if (g_) { dress_free_graph(g_); g_ = nullptr; }
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

// Standalone one-shot functions
inline DRESS::FitResult fit(int N, const std::vector<int>& U, const std::vector<int>& V,
                                   int maxIterations = 100, double epsilon = 1e-6) {
    DRESS d(N, U, V);
    return d.fit(maxIterations, epsilon);
}

inline DRESS::DeltaFitResult delta_fit(int N, const std::vector<int>& U, const std::vector<int>& V,
                                              int k = 0, int maxIterations = 100, double epsilon = 1e-6,
                                              int nSamples = 0, unsigned int seed = 0,
                                              bool keepMultisets = false, bool computeHistogram = true) {
    DRESS d(N, U, V);
    return d.deltaFit(k, maxIterations, epsilon, nSamples, seed, keepMultisets, computeHistogram);
}

inline DRESS::NablaFitResult nabla_fit(int N, const std::vector<int>& U, const std::vector<int>& V,
                                              int k = 0, int maxIterations = 100, double epsilon = 1e-6,
                                              int nSamples = 0, unsigned int seed = 0,
                                              bool keepMultisets = false, bool computeHistogram = true) {
    DRESS d(N, U, V);
    return d.nablaFit(k, maxIterations, epsilon, nSamples, seed, keepMultisets, computeHistogram);
}

} // namespace dress

#endif // DRESS_HPP
