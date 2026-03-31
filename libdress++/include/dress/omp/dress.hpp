#ifndef DRESS_OMP_HPP
#define DRESS_OMP_HPP

// OpenMP-parallel DRESS — import-based switching.
//
// Same API as the CPU DRESS class, but fit() parallelises edges
// and deltaFit() parallelises subgraphs.
//
//   // Sequential:
//   #include "dress/dress.hpp"
//   DRESS g(N, U, V, W);
//   g.fit(100, 1e-6);
//
//   // OpenMP (same API, different include):
//   #include "dress/omp/dress.hpp"
//   omp::DRESS g(N, U, V, W);
//   g.fit(100, 1e-6);

#include "dress/dress.hpp"
#include "dress/omp/dress_omp.h"

namespace dress {
namespace omp {

class DRESS : public dress::DRESS {
public:
    using dress::DRESS::DRESS;

    // OpenMP-parallel iterative dress fitting (edge-parallel).
    FitResult fit(int maxIterations, double epsilon) {
        if (!raw())
            throw std::logic_error("DRESS: accessing a moved-from or null graph");
        int    iters = 0;
        double d     = 0.0;
        ::dress_fit_omp(raw(), maxIterations, epsilon, &iters, &d);
        return {iters, d};
    }

    // OpenMP-parallel Δ^k-DRESS (subgraph-parallel).
    DeltaFitResult deltaFit(int k, int maxIterations, double epsilon,
                            int nSamples = 0, unsigned int seed = 0,
                            bool keepMultisets = false,
                            bool computeHistogram = true) {
        if (!raw())
            throw std::logic_error("DRESS: accessing a moved-from or null graph");
        int hsize = 0;
        int E = raw()->E;

        double *ms_ptr = nullptr;
        int64_t cnk = 0;
        dress_hist_pair_t *h = ::dress_delta_fit_omp_strided(raw(), k, maxIterations, epsilon,
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

    // OpenMP-parallel ∇^k-DRESS (tuple-parallel).
    NablaFitResult nablaFit(int k, int maxIterations, double epsilon,
                            int nSamples = 0, unsigned int seed = 0,
                            bool keepMultisets = false,
                            bool computeHistogram = true) {
        if (!raw())
            throw std::logic_error("DRESS: accessing a moved-from or null graph");
        int hsize = 0;
        int E = raw()->E;

        double *ms_ptr = nullptr;
        int64_t pnk = 0;
        dress_hist_pair_t *h = ::dress_nabla_fit_omp(raw(), k, maxIterations, epsilon,
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
};

// Standalone one-shot functions (OpenMP)
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

} // namespace omp
} // namespace dress

#endif // DRESS_OMP_HPP
