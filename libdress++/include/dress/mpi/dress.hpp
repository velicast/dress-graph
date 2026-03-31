#ifndef DRESS_MPI_HPP
#define DRESS_MPI_HPP

/**
 * MPI-distributed Δ^k-DRESS (C++ wrapper, CPU backend).
 *
 * Same API as the CPU DRESS class, but deltaFit distributes
 * subgraph enumeration across MPI ranks and reduces the results.
 *
 *   // CPU:
 *   #include "dress/dress.hpp"
 *   DRESS g(N, U, V, W, variant, precompute);
 *   auto dr = g.deltaFit(2, 100, 1e-6);
 *
 *   // MPI — same API, different include:
 *   #include "dress/mpi/dress.hpp"
 *   mpi::DRESS g(N, U, V, W, variant, precompute);
 *   auto dr = g.deltaFit(2, 100, 1e-6, false, MPI_COMM_WORLD);
 */

#include "dress/dress.hpp"
#include "dress/mpi/dress_mpi.h"
#include <mpi.h>

namespace dress {
namespace mpi {

class DRESS : public dress::DRESS {
private:
    using dress::DRESS::fit;  // MPI classes expose only deltaFit/nablaFit.

public:
    using dress::DRESS::DRESS;

    /**
     * MPI-distributed Δ^k-DRESS.
     *
     * Distributes the C(N,k) subgraph enumeration across MPI ranks
     * using stride-based partitioning, then reduces histograms and
     * (optionally) multiset matrices with MPI_Allreduce(SUM).
     *
     * @param k              Number of vertices to remove per subset.
     * @param maxIterations  Max DRESS iterations per subgraph.
     * @param epsilon        Convergence threshold / bin width.
     * @param keepMultisets  If true, return per-subgraph edge values.
     * @param comm           MPI communicator (default MPI_COMM_WORLD).
     */
    DeltaFitResult deltaFit(int k, int maxIterations, double epsilon,
                            int nSamples = 0, unsigned int seed = 0,
                            bool keepMultisets = false,
                            bool computeHistogram = true,
                            MPI_Comm comm = MPI_COMM_WORLD) {
        ensureValid();

        int hsize = 0;
        double *ms_ptr = nullptr;
        int64_t cnk = 0;

        dress_hist_pair_t *h = ::dress_delta_fit_mpi(
            raw(), k, maxIterations, epsilon,
        nSamples, seed,
        computeHistogram ? &hsize : nullptr,
        keepMultisets ? 1 : 0,
        keepMultisets ? &ms_ptr : nullptr,
        &cnk, comm);

        DeltaFitResult result;
        if (h && hsize > 0) {
            for (int i = 0; i < hsize; i++) {
                result.histogram.emplace_back(h[i].value, h[i].count);
            }
            std::free(h);
        }
        result.num_subgraphs = cnk;

        if (keepMultisets && ms_ptr) {
            int E = raw()->E;
            result.multisets.assign(ms_ptr, ms_ptr + cnk * E);
            std::free(ms_ptr);
        }
        return result;
    }

    // MPI-distributed ∇^k-DRESS.
    NablaFitResult nablaFit(int k, int maxIterations, double epsilon,
                            int nSamples = 0, unsigned int seed = 0,
                            bool keepMultisets = false,
                            bool computeHistogram = true,
                            MPI_Comm comm = MPI_COMM_WORLD) {
        ensureValid();

        int hsize = 0;
        double *ms_ptr = nullptr;
        int64_t pnk = 0;

        dress_hist_pair_t *h = ::dress_nabla_fit_mpi(
            raw(), k, maxIterations, epsilon,
        nSamples, seed,
        computeHistogram ? &hsize : nullptr,
        keepMultisets ? 1 : 0,
        keepMultisets ? &ms_ptr : nullptr,
        &pnk, comm);

        NablaFitResult result;
        if (h && hsize > 0) {
            for (int i = 0; i < hsize; i++) {
                result.histogram.emplace_back(h[i].value, h[i].count);
            }
            std::free(h);
        }
        result.num_tuples = pnk;

        if (keepMultisets && ms_ptr) {
            int E = raw()->E;
            result.multisets.assign(ms_ptr, ms_ptr + pnk * E);
            std::free(ms_ptr);
        }
        return result;
    }
};

// Standalone one-shot functions (MPI)
inline DRESS::DeltaFitResult delta_fit(int N, const std::vector<int>& U, const std::vector<int>& V,
                                              int k = 0, int maxIterations = 100, double epsilon = 1e-6,
                                              int nSamples = 0, unsigned int seed = 0,
                                              bool keepMultisets = false, bool computeHistogram = true,
                                              MPI_Comm comm = MPI_COMM_WORLD) {
    DRESS d(N, U, V);
    return d.deltaFit(k, maxIterations, epsilon, nSamples, seed, keepMultisets, computeHistogram, comm);
}

inline DRESS::NablaFitResult nabla_fit(int N, const std::vector<int>& U, const std::vector<int>& V,
                                              int k = 0, int maxIterations = 100, double epsilon = 1e-6,
                                              int nSamples = 0, unsigned int seed = 0,
                                              bool keepMultisets = false, bool computeHistogram = true,
                                              MPI_Comm comm = MPI_COMM_WORLD) {
    DRESS d(N, U, V);
    return d.nablaFit(k, maxIterations, epsilon, nSamples, seed, keepMultisets, computeHistogram, comm);
}

} // namespace mpi
} // namespace dress

#endif // DRESS_MPI_HPP
