#ifndef MATMUL_H
#define MATMUL_H
#include <vector>

void matmul(const std::vector<float>& A,
            const std::vector<float>& B,
            std::vector<float>& C,
            int M, int K, int N);
#endif
