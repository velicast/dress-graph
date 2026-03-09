#ifndef DRESS_CUDA_HPP
#define DRESS_CUDA_HPP

// GPU-accelerated DRESS — import-based switching.
//
// Same API as the CPU DRESS class, but fit() and deltaFit() run on the GPU.
//
//   // CPU:
//   #include "dress/dress.hpp"
//   DRESS g(N, U, V, W, variant, precompute);
//   auto [iters, delta] = g.fit(100, 1e-6);
//
//   // CUDA (same API, different include):
//   #include "dress/cuda/dress.hpp"
//   cuda::DRESS g(N, U, V, W, variant, precompute);
//   auto [iters, delta] = g.fit(100, 1e-6);

#include "dress/dress.hpp"
#include "dress/cuda/dress_cuda.h"

namespace cuda {

class DRESS : public ::DRESS {
public:
    // Inherit all constructors from the CPU DRESS class.
    using ::DRESS::DRESS;

    // GPU-accelerated iterative dress fitting.
    // Shadows the CPU DRESS::fit() — identical signature and return type.
    FitResult fit(int maxIterations, double epsilon) {
        if (!raw())
            throw std::logic_error("DRESS: accessing a moved-from or null graph");
        int    iters = 0;
        double d     = 0.0;
        ::dress_fit_cuda(raw(), maxIterations, epsilon, &iters, &d);
        return {iters, d};
    }

    // GPU-accelerated Δ^k-DRESS.
    // Shadows the CPU DRESS::deltaFit() — identical signature and return type.
    DeltaFitResult deltaFit(int k, int maxIterations, double epsilon,
                            bool keepMultisets = false,
                            int offset = 0, int stride = 1) {
        if (!raw())
            throw std::logic_error("DRESS: accessing a moved-from or null graph");
        int hsize = 0;
        int E = raw()->E;

        double *ms_ptr = nullptr;
        int64_t cnk = 0;
        int64_t *h = ::delta_dress_fit_cuda_strided(raw(), k, maxIterations, epsilon,
                                            &hsize,
                                            keepMultisets ? 1 : 0,
                                            keepMultisets ? &ms_ptr : nullptr,
                                            &cnk,
                                            offset, stride);
        if (!h) throw std::runtime_error("DRESS: delta_dress_fit_cuda returned NULL");

        DeltaFitResult result;
        result.hist_size = hsize;
        result.histogram.assign(h, h + hsize);
        result.num_subgraphs = cnk;
        if (keepMultisets && ms_ptr) {
            result.multisets.assign(ms_ptr, ms_ptr + cnk * E);
            std::free(ms_ptr);
        }
        std::free(h);
        return result;
    }
};

} // namespace cuda

#endif // DRESS_CUDA_HPP
