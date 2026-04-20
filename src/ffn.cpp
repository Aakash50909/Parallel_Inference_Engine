#include "ffn.h"
#include "matmul.h"
#include <algorithm>
#include <stdexcept>
void ffn(const std::vector<float>& x,
         const std::vector<float>& W1,
         const std::vector<float>& b1,
         const std::vector<float>& W2,
         const std::vector<float>& b2,
         std::vector<float>& out,
         int seq_len, int d_model, int d_ff) {
if ((int)W1.size() != d_model * d_ff)
        throw std::invalid_argument("ffn: W1 size mismatch");
    if ((int)b1.size() != d_ff)
        throw std::invalid_argument("ffn: b1 size mismatch");
    if ((int)W2.size() != d_ff * d_model)
        throw std::invalid_argument("ffn: W2 size mismatch");
    if ((int)b2.size() != d_model)
        throw std::invalid_argument("ffn: b2 size mismatch");
// Step 1: hidden = x * W1  shape: (seq_len x d_ff)
    std::vector<float> hidden;
    matmul(x, W1, hidden, seq_len, d_model, d_ff);
// Step 2: add bias b1 and apply ReLU
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_ff; j++) {
            hidden[i * d_ff + j] += b1[j];
hidden[i * d_ff + j] = std::max(0.0f, hidden[i * d_ff + j]);
}
    }
// Step 3: out = hidden * W2  shape: (seq_len x d_model)
    matmul(hidden, W2, out, seq_len, d_ff, d_model);
// Step 4: add bias b2
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d_model; j++)
            out[i * d_model + j] += b2[j];
}

