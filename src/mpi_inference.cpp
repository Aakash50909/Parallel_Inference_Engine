#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "matmul.h"
#include "attention.h"
#include "ffn.h"
#include "layernorm.h"

// Fills a vector with deterministic random floats
// Same seed = same data on every rank, important for consistency
void rand_fill(std::vector<float>& v, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (float& x : v) x = dist(rng);
}

// Simulates one transformer block: layernorm -> attention -> layernorm -> ffn
// This is what each rank runs on its assigned layers
void transformer_block(std::vector<float>& x,
                        int seq_len, int d_model, int d_ff) {
    int d_k = d_model;
    int d_v = d_model;

    // Layernorm before attention (pre-norm style, like GPT-2)
    std::vector<float> gamma(d_model, 1.0f);
    std::vector<float> beta(d_model, 0.0f);
    layernorm(x, gamma, beta, seq_len, d_model);

    // Random projection weights for Q, K, V
    // In a real model these would be loaded from a checkpoint
    std::vector<float> Wq(d_model * d_k);
    std::vector<float> Wk(d_model * d_k);
    std::vector<float> Wv(d_model * d_v);
    rand_fill(Wq, 1); rand_fill(Wk, 2); rand_fill(Wv, 3);

    std::vector<float> Q, K, V;
    matmul(x, Wq, Q, seq_len, d_model, d_k);
    matmul(x, Wk, K, seq_len, d_model, d_k);
    matmul(x, Wv, V, seq_len, d_model, d_v);

    std::vector<float> attn_out;
    attention(Q, K, V, attn_out, seq_len, d_k, d_v);

    // Residual connection: x = x + attention(x)
    for (int i = 0; i < seq_len * d_model; i++)
        x[i] += attn_out[i];

    // Layernorm before FFN
    layernorm(x, gamma, beta, seq_len, d_model);

    // FFN weights
    std::vector<float> W1(d_model * d_ff);
    std::vector<float> W2(d_ff * d_model);
    std::vector<float> b1(d_ff, 0.0f);
    std::vector<float> b2(d_model, 0.0f);
    rand_fill(W1, 4); rand_fill(W2, 5);

    std::vector<float> ffn_out;
    ffn(x, W1, b1, W2, b2, ffn_out, seq_len, d_model, d_ff);

    // Residual connection: x = x + ffn(x)
    for (int i = 0; i < seq_len * d_model; i++)
        x[i] += ffn_out[i];
}

int main(int argc, char** argv) {
    // MPI_Init must be the first MPI call in every program
    // It sets up the communication infrastructure between all processes
    MPI_Init(&argc, &argv);

    int rank, size;
    // MPI_Comm_rank tells this process its own ID (0, 1, 2...)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size tells this process how many total processes exist
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0)
            std::cerr << "This program requires exactly 2 MPI ranks\n";
        MPI_Finalize();
        return 1;
    }

    // Model hyperparameters — small but realistic proportions
    const int seq_len = 8;    // number of tokens
    const int d_model = 64;   // embedding dimension
    const int d_ff    = 256;  // FFN hidden dim (4x d_model)
    const int n_layers = 12;  // total transformer layers
    const int layers_per_rank = n_layers / 2; // 6 layers each

    // Size of the activation tensor passed between ranks
    // shape: (seq_len x d_model), flat float array
    int activation_size = seq_len * d_model;

    std::vector<float> x(activation_size);

    auto t_start = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        // Rank 0: initialize input and run first 6 layers
        rand_fill(x, 99);
        std::cout << "[Rank 0] Starting pipeline — running layers 1-"
                  << layers_per_rank << "\n";

        for (int layer = 0; layer < layers_per_rank; layer++)
            transformer_block(x, seq_len, d_model, d_ff);

        std::cout << "[Rank 0] Layers done. Sending activations to Rank 1...\n";

        // MPI_Send(buffer, count, datatype, destination, tag, communicator)
        // Blocking send — rank 0 waits here until rank 1 has received the data
        MPI_Send(x.data(), activation_size, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);

        std::cout << "[Rank 0] Send complete.\n";

    } else {
        // Rank 1: receive activations from rank 0, run last 6 layers
        std::cout << "[Rank 1] Waiting to receive activations from Rank 0...\n";

        // MPI_Recv(buffer, count, datatype, source, tag, communicator, status)
        // Blocking recv — rank 1 waits here until data arrives from rank 0
        MPI_Recv(x.data(), activation_size, MPI_FLOAT, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::cout << "[Rank 1] Received activations. Running layers "
                  << layers_per_rank + 1 << "-" << n_layers << "\n";

        for (int layer = 0; layer < layers_per_rank; layer++)
            transformer_block(x, seq_len, d_model, d_ff);

        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t_end - t_start;

        std::cout << "[Rank 1] Pipeline complete.\n";
        std::cout << "Total pipeline time: " << elapsed.count() << " ms\n";
        std::cout << "Output norm (sanity check): ";

        // Print L2 norm of output as a sanity check
        // If this is NaN or 0, something went wrong in the pipeline
        float norm = 0.0f;
        for (float v : x) norm += v * v;
        std::cout << sqrtf(norm) << "\n";
    }

    // MPI_Finalize must be the last MPI call — cleans up all resources
    MPI_Finalize();
    return 0;
}
