#include "benchmark.h"
#include <iostream>
#include <iomanip>

double time_ms(std::function<void()> fn, int trials) {
    // Warmup run — not counted
    // First call often slower due to cache misses and branch predictor warmup
    fn();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; i++)
        fn();
    auto end = std::chrono::high_resolution_clock::now();

    // Returns average time per call in milliseconds
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / trials;
}

void print_result(const std::string& label, double ms, double baseline_ms) {
    double speedup = baseline_ms / ms;
    std::cout << std::left  << std::setw(25) << label
              << std::right << std::setw(10) << std::fixed << std::setprecision(3) << ms << " ms"
              << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x\n";
}
