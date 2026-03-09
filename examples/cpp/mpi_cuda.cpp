// mpi_cuda.cpp — Rook vs Shrikhande with Δ¹-DRESS (MPI + CUDA, C++ wrapper)
// Keeps multisets and compares them to guarantee distinguishability.
//
// Build & run:
//   mpicxx -O2 -std=c++17 -o mpi_cuda mpi_cuda.cpp -ldress -ldress_cuda -lm
//   mpirun -np 4 ./mpi_cuda
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "dress/mpi/cuda/dress.hpp"

static std::vector<int> rook_s = {0,1,0,4,0,2,0,8,0,3,0,12,1,5,1,2,1,9,1,3,1,13,2,6,2,10,2,3,2,14,3,7,3,11,3,15,4,5,4,6,4,8,4,7,4,12,5,6,5,9,5,7,5,13,6,10,6,7,6,14,7,11,7,15,8,9,8,10,8,11,8,12,9,10,9,11,9,13,10,11,10,14,11,15,12,13,12,14,12,15,13,14,13,15,14,15};
static std::vector<int> rook_t = {1,0,4,0,2,0,8,0,3,0,12,0,5,1,2,1,9,1,3,1,13,1,6,2,10,2,3,2,14,2,7,3,11,3,15,3,5,4,6,4,8,4,7,4,12,4,6,5,9,5,7,5,13,5,10,6,7,6,14,6,11,7,15,7,9,8,10,8,11,8,12,8,10,9,11,9,13,9,11,10,14,10,15,11,13,12,14,12,15,12,14,13,15,13,15,14};

static std::vector<int> shri_s = {0,4,0,12,0,1,0,3,0,5,0,15,1,5,1,13,1,2,1,6,1,12,2,6,2,14,2,3,2,7,2,13,3,7,3,15,3,4,3,14,4,8,4,5,4,7,4,9,5,9,5,6,5,10,6,10,6,7,6,11,7,11,7,8,8,12,8,9,8,11,8,13,9,13,9,10,9,14,10,14,10,11,10,15,11,15,11,12,12,13,12,15,13,14,14,15};
static std::vector<int> shri_t = {4,0,12,0,1,0,3,0,5,0,15,0,5,1,13,1,2,1,6,1,12,1,6,2,14,2,3,2,7,2,13,2,7,3,15,3,4,3,14,3,8,4,5,4,7,4,9,4,9,5,6,5,10,5,10,6,7,6,11,6,11,7,8,7,12,8,9,8,11,8,13,8,13,9,10,9,14,9,14,10,11,10,15,10,15,11,12,11,13,12,15,12,14,13,15,14};

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    mpi::cuda::DRESS rook(16, rook_s, rook_t);
    mpi::cuda::DRESS shri(16, shri_s, shri_t);

    auto dr = rook.deltaFit(1, 100, 1e-6, true);
    auto ds = shri.deltaFit(1, 100, 1e-6, true);

    if (rank == 0) {
        std::cout << "Rook:       " << dr.hist_size << " bins, "
                  << dr.num_subgraphs << " subgraphs\n";
        std::cout << "Shrikhande: " << ds.hist_size << " bins, "
                  << ds.num_subgraphs << " subgraphs\n";
        std::cout << "Histograms differ:  "
                  << (dr.histogram != ds.histogram ? "yes" : "no") << "\n";

        // Canonicalize multisets: sort each row, then sort rows
        auto canonicalize = [](std::vector<double> &ms, int64_t ns, int E) {
            for (int64_t i = 0; i < ns; i++) {
                auto b = ms.begin() + i * E, e = b + E;
                std::sort(b, e, [](double a, double b) {
                    if (std::isnan(a) && std::isnan(b)) return false;
                    if (std::isnan(a)) return false;
                    if (std::isnan(b)) return true;
                    return a < b;
                });
            }
            std::vector<std::vector<double>> rows(ns);
            for (int64_t i = 0; i < ns; i++)
                rows[i].assign(ms.begin() + i * E, ms.begin() + i * E + E);
            std::sort(rows.begin(), rows.end());
            for (int64_t i = 0; i < ns; i++)
                std::copy(rows[i].begin(), rows[i].end(), ms.begin() + i * E);
        };

        int E = 96;
        canonicalize(dr.multisets, dr.num_subgraphs, E);
        canonicalize(ds.multisets, ds.num_subgraphs, E);

        bool ms_same = (dr.num_subgraphs == ds.num_subgraphs);
        if (ms_same) {
            for (size_t i = 0; i < dr.multisets.size(); i++) {
                if (std::isnan(dr.multisets[i]) && std::isnan(ds.multisets[i])) continue;
                if (dr.multisets[i] != ds.multisets[i]) { ms_same = false; break; }
            }
        }
        std::cout << "Multisets differ:   " << (ms_same ? "no" : "yes") << "\n";
    }
    MPI_Finalize();
}
