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

namespace mpi {

class DRESS : public ::DRESS {
public:
    using ::DRESS::DRESS;

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
                            bool keepMultisets = false,
                            MPI_Comm comm = MPI_COMM_WORLD) {
        ensureValid();

        int hsize = 0;
        double *ms_ptr = nullptr;
        int64_t cnk = 0;

        int64_t *h = ::delta_dress_fit_mpi(
            raw(), k, maxIterations, epsilon, &hsize,
            keepMultisets ? 1 : 0,
            keepMultisets ? &ms_ptr : nullptr,
            &cnk, comm);

        if (!h) throw std::runtime_error("DRESS: delta_dress_fit_mpi returned NULL");

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

} // namespace mpi

#endif // DRESS_MPI_HPP
