#include <iostream>
#include <vector>
#include "ffn.h"

int main() {
    // 2 tokens, d_model=4, d_ff=8 (2x expansion for test, real GPT uses 4x)
    int seq_len = 2, d_model = 4, d_ff = 8;
// Input: 2 tokens each with 4 features
    std::vector<float> x = {
        1, 0, 1, 0,
        0, 1, 0, 1
    };
// W1: (4 x 8) — all 0.5 for easy hand-verification
    std::vector<float> W1(d_model * d_ff, 0.5f);
    std::vector<float> b1(d_ff, 0.1f);
// W2: (8 x 4) — all 0.5
    std::vector<float> W2(d_ff * d_model, 0.5f);
    std::vector<float> b2(d_model, 0.0f);
std::vector<float> out;
    ffn(x, W1, b1, W2, b2, out, seq_len, d_model, d_ff);
std::cout << "FFN output (" << seq_len << " x " << d_model << "):\n";
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++)
            std::cout << out[i * d_model + j] << " ";
        std::cout << "\n";
    }
// Each output value should be 8 * 1.1 * 0.5 = 4.4
    bool passed = true;
    for (float v : out)
        if (std::abs(v - 4.4f) > 1e-3f) passed = false;

    std::cout << "FFN test " << (passed ? "PASSED" : "FAILED") << "\n";
    return 0;
}
