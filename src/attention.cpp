#include "attention.h"
#include "matmul.h"
#include "softmax.h"
#include <cmath>
#include <vector>

void attention(const std::vector<float>& Q,
               const std::vector<float>& K,
               const std::vector<float>& V,
               std::vector<float>& out,
               int seq_len, int d_k, int d_v) {

    std::vector<float> Kt(d_k * seq_len);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d_k; j++)
            Kt[j * seq_len + i] = K[i * d_k + j];

    std::vector<float> scores;
    matmul(Q, Kt, scores, seq_len, d_k, seq_len);

    // Pre-allocate output before entering parallel region
    // All threads write to different rows so no mutex needed
    out.assign(seq_len * d_v, 0.0f);

    float scale = 1.0f / sqrtf((float)d_k);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < seq_len; i++) {

        for (int j = 0; j < seq_len; j++)
            scores[i * seq_len + j] *= scale;

        float max_val = scores[i * seq_len];
        for (int j = 1; j < seq_len; j++)
            if (scores[i * seq_len + j] > max_val)
                max_val = scores[i * seq_len + j];

        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
            sum += scores[i * seq_len + j];
        }
        for (int j = 0; j < seq_len; j++)
            scores[i * seq_len + j] /= sum;

        for (int j = 0; j < d_v; j++) {
            float val = 0.0f;
            for (int k = 0; k < seq_len; k++)
                val += scores[i * seq_len + k] * V[k * d_v + j];
            out[i * d_v + j] = val;
        }
    }
}
