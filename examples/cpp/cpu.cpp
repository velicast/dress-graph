// cpu.cpp — Prism vs K₃,₃ with DRESS (CPU, C++ wrapper)
//
// Build:
//   g++ -O2 -std=c++17 -o cpu cpu.cpp -ldress -lm
#include <iostream>
#include <algorithm>
#include <vector>
#include "dress/dress.hpp"
using namespace dress;

int main() {
    // Prism (C₃ □ K₂)
    std::vector<int> pU = {0,1,1,2,2,0,0,3,1,4,2,5,3,4,4,5,5,3};
    std::vector<int> pV = {1,0,2,1,0,2,3,0,4,1,5,2,4,3,5,4,3,5};
    // K₃,₃
    std::vector<int> kU = {0,3,0,4,0,5,1,3,1,4,1,5,2,3,2,4,2,5};
    std::vector<int> kV = {3,0,4,0,5,0,3,1,4,1,5,1,3,2,4,2,5,2};

    DRESS prism(6, pU, pV);
    DRESS k33(6, kU, kV);

    prism.fit(100, 1e-6);
    k33.fit(100, 1e-6);

    auto fp = [](DRESS& g) {
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
