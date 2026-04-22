#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include "matmul.h"
#include "attention.h"
#include "ffn.h"
#include "layernorm.h"

void rand_fill(std::vector<float>& v, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (float& x : v) x = dist(rng);
}

void transformer_block(std::vector<float>& x,
                       int seq_len, int d_model, int d_ff) {
    std::vector<float> gamma(d_model, 1.0f);
    std::vector<float> beta(d_model, 0.0f);
    layernorm(x, gamma, beta, seq_len, d_model);

    std::vector<float> Wq(d_model * d_model);
    std::vector<float> Wk(d_model * d_model);
    std::vector<float> Wv(d_model * d_model);
    rand_fill(Wq, 1); rand_fill(Wk, 2); rand_fill(Wv, 3);

    std::vector<float> Q, K, V;
    matmul(x, Wq, Q, seq_len, d_model, d_model);
    matmul(x, Wk, K, seq_len, d_model, d_model);
    matmul(x, Wv, V, seq_len, d_model, d_model);

    std::vector<float> attn_out;
    attention(Q, K, V, attn_out, seq_len, d_model, d_model);

    for (int i = 0; i < seq_len * d_model; i++)
        x[i] += attn_out[i];

    layernorm(x, gamma, beta, seq_len, d_model);

    std::vector<float> W1(d_model * d_ff);
    std::vector<float> W2(d_ff * d_model);
    std::vector<float> b1(d_ff, 0.0f);
    std::vector<float> b2(d_model, 0.0f);
    rand_fill(W1, 4); rand_fill(W2, 5);

    std::vector<float> ffn_out;
    ffn(x, W1, b1, W2, b2, ffn_out, seq_len, d_model, d_ff);

    for (int i = 0; i < seq_len * d_model; i++)
        x[i] += ffn_out[i];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int seq_len  = 8;
    const int d_model  = 64;
    const int d_ff     = 256;
    const int n_layers = 12;
    const int trials   = 5;
    int activation_size = seq_len * d_model;

    // Divide layers as evenly as possible across all ranks
    // e.g. 12 layers / 3 ranks = 4 layers each
    // If it doesn't divide evenly, rank 0 gets the remainder
    int layers_per_rank = n_layers / size;
    int remainder       = n_layers % size;
    // rank 0 handles any leftover layers
    int my_layers = (rank == 0) ? layers_per_rank + remainder
                                : layers_per_rank;

    std::vector<float> x(activation_size);

    // --- Serial baseline: rank 0 runs all layers, others idle ---
    double serial_ms = 0.0;
    if (rank == 0) {
        rand_fill(x, 99);
        for (int l = 0; l < n_layers; l++)
            transformer_block(x, seq_len, d_model, d_ff);

        double total = 0.0;
        for (int t = 0; t < trials; t++) {
            rand_fill(x, 99);
            double t0 = MPI_Wtime();
            for (int l = 0; l < n_layers; l++)
                transformer_block(x, seq_len, d_model, d_ff);
            total += (MPI_Wtime() - t0) * 1000.0;
        }
        serial_ms = total / trials;
    }
    // Share serial time with all ranks so anyone can print it
    MPI_Bcast(&serial_ms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- Pipeline benchmark ---
    // Each rank receives from its left neighbour, computes, sends right
    // Rank 0 has no left neighbour (it starts the pipeline)
    // Last rank has no right neighbour (it ends the pipeline)
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int t = 0; t < trials; t++) {
        if (rank == 0) {
            // First rank: initialize input and run its layers
            rand_fill(x, 99);
            for (int l = 0; l < my_layers; l++)
                transformer_block(x, seq_len, d_model, d_ff);

            // Send activations to the next rank in the pipeline
            // Tag = 0, destination = rank + 1
            MPI_Send(x.data(), activation_size, MPI_FLOAT,
                     rank + 1, 0, MPI_COMM_WORLD);

        } else if (rank == size - 1) {
            // Last rank: receive from previous rank, run its layers
            // Source = rank - 1, tag = 0
            MPI_Recv(x.data(), activation_size, MPI_FLOAT,
                     rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int l = 0; l < my_layers; l++)
                transformer_block(x, seq_len, d_model, d_ff);

            // No send — this is the end of the pipeline

        } else {
            // Middle ranks: receive from left, compute, send right
            // This is where true pipeline overlap would happen on a cluster
            MPI_Recv(x.data(), activation_size, MPI_FLOAT,
                     rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int l = 0; l < my_layers; l++)
                transformer_block(x, seq_len, d_model, d_ff);

            MPI_Send(x.data(), activation_size, MPI_FLOAT,
                     rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double pipeline_ms = (MPI_Wtime() - t_start) * 1000.0 / trials;

    // Only the last rank prints — it finishes last so its time = total time
    if (rank == size - 1) {
        double speedup = serial_ms / pipeline_ms;
        std::cout << "\n=== MPI Pipeline Benchmark (" << size << " ranks) ===\n";
        std::cout << std::left  << std::setw(28) << "Configuration"
                  << std::right << std::setw(12) << "Time (ms)"
                  << std::setw(10) << "Speedup\n";
        std::cout << std::string(50, '-') << "\n";
        std::cout << std::left  << std::setw(28) << "Serial (1 rank)"
                  << std::right << std::setw(12) << std::fixed
                  << std::setprecision(3) << serial_ms
                  << std::setw(9) << "1.00x\n";
        std::cout << std::left  << std::setw(28)
                  << ("MPI pipeline (" + std::to_string(size) + " ranks)")
                  << std::right << std::setw(12) << std::fixed
                  << std::setprecision(3) << pipeline_ms
                  << std::setw(9) << std::setprecision(2) << speedup << "x\n";
        std::cout << "\nLayers per rank: ~" << layers_per_rank
                  << " (" << n_layers << " total / " << size << " ranks)\n";
    }

    MPI_Finalize();
    return 0;
}
