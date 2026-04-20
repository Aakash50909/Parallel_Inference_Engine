#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <string>
#include "matmul.h"
#include "benchmark.h"

void rand_fill(std::vector<float>& v) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) x = dist(rng);
}

int main() {
    std::cout << std::left  << std::setw(25) << "Configuration"
              << std::right << std::setw(12) << "Time"
              << std::setw(10) << "Speedup\n";
    std::cout << std::string(47, '-') << "\n";

    for (int N : {64, 128, 256, 512, 1024}) {
        std::vector<float> A(N * N), B(N * N), C;
        rand_fill(A);
        rand_fill(B);

        std::string label = "serial " + std::to_string(N) + "x" + std::to_string(N);

        double ms = time_ms([&]() {
            matmul(A, B, C, N, N, N);
        });

        print_result(label, ms, ms);
    }

    return 0;
}
