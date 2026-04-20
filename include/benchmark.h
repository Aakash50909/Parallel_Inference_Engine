#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <chrono>
#include <string>
#include <functional>

// Runs a function `trials` times and returns average time in milliseconds
double time_ms(std::function<void()> fn, int trials = 5);

// Prints a formatted benchmark result line
void print_result(const std::string& label, double ms, double baseline_ms);

#endif // BENCHMARK_H
