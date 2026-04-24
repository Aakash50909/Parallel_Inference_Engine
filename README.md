# Parallelizing Transformer Inference: A Hybrid Approach

**Author:** Aakash Chandra

## Overview
This project implements the complete forward pass of a GPT-style Large Language Model (LLM) entirely from scratch in C++. It is built without reliance on external machine learning libraries such as PyTorch, BLAS, or cuDNN. The primary goal is to accelerate compute-intensive General Matrix-Matrix Multiplication (GEMM) and attention mechanisms using multiple parallel computing paradigms.

---

## 📄 Read the Full Report
*(To embed the PDF directly in your GitHub README, ensure `Report_102317242.pdf` is uploaded to your repository and use the code below. Note: GitHub's native markdown viewer sometimes restricts direct PDF embedding, so a direct link is also provided as a fallback).*

<details>
<summary><b>View Full PDF Report</b></summary>
<object data="Report_102317242.pdf" type="application/pdf" width="100%" height="800px">
  <p>Unable to display PDF file. <a href="Report_102317242.pdf">Download instead</a>.</p>
</object>
</details>

---

## Project Phases & Architecture
The inference engine is divided into five distinct phases to analyze and optimize computational throughput:
* **Phase I (Serial Baseline):** A standard C++17 implementation of all kernels (GEMM, Softmax, LayerNorm, Attention) acting as the baseline.
* **Phase II (OpenMP):** Shared memory parallelism utilizing OpenMP loop-level directives across CPU cores.
* **Phase III (MPI):** Distributed memory parallelism simulating a pipeline by decomposing transformer layers across multiple MPI ranks.
* **Phase IV (CUDA):** Heterogeneous GPU computing with a custom CUDA kernel featuring 16x16 shared memory tiling to reduce global memory bandwidth pressure.
* **Phase V (Hybrid Engine):** A unified pipeline combining MPI, OpenMP, and CUDA for maximum model-parallel throughput.

## Performance Highlights
Benchmarks were conducted using an AMD Ryzen CPU (8 physical cores) and an NVIDIA RTX 3050 Laptop GPU (2048 CUDA cores). 

* **OpenMP:** Achieved a peak speedup of 9.13x at 8 threads on 512x512 matrices.
* **MPI Pipeline:** Achieved a peak speedup of 2.95x at 6 ranks.
* **CUDA GPU:** Delivered a peak speedup of 773.93x for 2048x2048 matrices utilizing shared memory tiling.
* **Hybrid (MPI + OpenMP + CUDA):** Achieved the highest peak speedup of 848.60x for 2048x2048 matrices (4 batches), surpassing standalone CUDA performance.

## Getting Started

### Prerequisites
* **C++ Compiler:** Supporting C++17.
* **OpenMP:** For CPU multi-threading.
* **OpenMPI:** For pipeline distribution.
* **CUDA Toolkit (nvcc):** For GPU acceleration.

### Build Instructions
The hybrid inference engine can be compiled using `nvcc` by linking the respective backends:

```bash
nvcc -std=c++17 -O3 -ccbin mpicxx -Xcompiler -fopenmp \
src/matmul.cpp src/matmul_cuda.cu src/hybrid_main.cpp \
-o hybrid_engine

mpirun -np 2 ./hybrid_engine
