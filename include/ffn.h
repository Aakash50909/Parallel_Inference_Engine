#ifndef FFN_H
#define FFN_H

#include <vector>

void ffn(const std::vector<float>& x,
         const std::vector<float>& W1,
         const std::vector<float>& b1,
         const std::vector<float>& W2,
         const std::vector<float>& b2,
         std::vector<float>& out,
         int seq_len, int d_model, int d_ff);

#endif // FFN_H
