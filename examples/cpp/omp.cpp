// omp.cpp — Prism vs K₃,₃ with DRESS (OpenMP, C++ wrapper)
//
// Build:
//   g++ -O2 -std=c++17 -fopenmp -o omp omp.cpp -ldress -lm
#include <iostream>
#include <algorithm>
#include <vector>
#include "dress/omp/dress.hpp"
using namespace dress;

int main() {
    std::vector<int> pU = {0,1,1,2,2,0,0,3,1,4,2,5,3,4,4,5,5,3};
    std::vector<int> pV = {1,0,2,1,0,2,3,0,4,1,5,2,4,3,5,4,3,5};
    std::vector<int> kU = {0,3,0,4,0,5,1,3,1,4,1,5,2,3,2,4,2,5};
    std::vector<int> kV = {3,0,4,0,5,0,3,1,4,1,5,1,3,2,4,2,5,2};

    omp::DRESS prism(6, pU, pV);
    omp::DRESS k33(6, kU, kV);

    prism.fit(100, 1e-6);
    k33.fit(100, 1e-6);

    auto fp = [](omp::DRESS& g) {
        std::vector<double> v(g.edgeDressValues(),
                              g.edgeDressValues() + g.numEdges());
        std::sort(v.begin(), v.end());
        return v;
    };

    auto a = fp(prism), b = fp(k33);
    std::cout << "Prism: ";
    for (auto d : a) std::cout << d << " ";
    std::cout << "\nK3,3:  ";
    for (auto d : b) std::cout << d << " ";
    std::cout << "\nDistinguished: " << (a != b ? "yes" : "no") << "\n";
}
