#ifndef DRESS_MPI_CUDA_HPP
#define DRESS_MPI_CUDA_HPP

/**
 * MPI-distributed Δ^k-DRESS (C++ wrapper, CUDA backend).
 *
 * Same API as the CUDA DRESS class, but deltaFit distributes
 * subgraph enumeration across MPI ranks and reduces the results.
 *
 *   // CUDA:
 *   #include "dress/cuda/dress.hpp"
 *   cuda::DRESS g(N, U, V, W, variant, precompute);
 *   auto dr = g.deltaFit(2, 100, 1e-6);
 *
 *   // MPI + CUDA — same API, different include:
 *   #include "dress/mpi/cuda/dress.hpp"
 *   mpi::cuda::DRESS g(N, U, V, W, variant, precompute);
 *   auto dr = g.deltaFit(2, 100, 1e-6, false, MPI_COMM_WORLD);
 */

#include "dress/cuda/dress.hpp"
#include "dress/mpi/dress_mpi.h"
#include <mpi.h>

namespace mpi {
namespace cuda {

class DRESS : public ::cuda::DRESS {
public:
    using ::cuda::DRESS::DRESS;

    /**
     * MPI-distributed Δ^k-DRESS (CUDA + MPI).
     *
     * Each rank runs GPU-accelerated DRESS on its stride of subgraphs,
     * then reduces histograms and multisets across the communicator.
     *
     * @param k              Number of vertices to remove per subset.
     * @param maxIterations  Max DRESS iterations per subgraph.
     * @param epsilon        Convergence threshold / bin width.
     * @param keepMultisets  If true, return per-subgraph edge values.
     * @param comm           MPI communicator (default MPI_COMM_WORLD).
     */
    DeltaFitResult deltaFit(int k, int maxIterations, double epsilon,
                            bool keepMultisets = false,
                            MPI_Comm comm = MPI_COMM_WORLD) {
        if (!raw())
            throw std::logic_error("DRESS: accessing a moved-from or null graph");

        int hsize = 0;
        double *ms_ptr = nullptr;
        int64_t cnk = 0;

        int64_t *h = ::delta_dress_fit_mpi_cuda(
            raw(), k, maxIterations, epsilon, &hsize,
            keepMultisets ? 1 : 0,
            keepMultisets ? &ms_ptr : nullptr,
            &cnk, comm);

        if (!h) throw std::runtime_error("DRESS: delta_dress_fit_mpi_cuda returned NULL");

        DeltaFitResult result;
        result.hist_size = hsize;
        result.histogram.assign(h, h + hsize);
        result.num_subgraphs = cnk;

        if (keepMultisets && ms_ptr) {
            int E = raw()->E;
            result.multisets.assign(ms_ptr, ms_ptr + cnk * E);
            std::free(ms_ptr);
        }

        std::free(h);
        return result;
    }
};

} // namespace cuda
} // namespace mpi

#endif // DRESS_MPI_CUDA_HPP
