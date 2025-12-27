# High-Performance GPU Implementation of WENO5 with TVD-RK3

## 1. Overview
This project implements a high-order **WENO5** (Weighted Essentially Non-Oscillatory) scheme to solve 1D hyperbolic conservation laws ($u_t + f(u)_x = 0$). To ensure time-stepping stability and maintain high-order accuracy, a **3rd-order TVD Runge-Kutta (RK3)** method is employed.

The core contribution of this project is the progressive optimization of the CUDA kernel, moving from a standard global memory approach to register-level communication and thread coarsening.

## 2. Methodology & Optimization Stages

### V1: Naive Baseline (Global Memory Bound)
* **Strategy**: Each thread calculates one grid point $u_i$ by loading its 6-point stencil ($u_{i-3}$ to $u_{i+2}$) directly from global memory.
* **Bottleneck**: High redundant memory traffic. Each element is read 6 times across different threads, leading to severe memory bandwidth saturation.

### V2: Warp Shuffle Optimization (Communication Optimized)
* **Strategy**: Leverages the **Memory Hierarchy** by using `__shfl_sync` primitives. Each thread loads only **one** element into a register and "shuffles" it to its neighbors within the same warp.
* **Benefit**: Reduces global memory loads from 6 per thread to approximately **1.15 per thread** ($(32+5)/32$).
* **Insight**: This approach bypasses Shared Memory, utilizing the fastest register-to-register communication path available in NVIDIA GPUs.

### V3: Register Tiling (Instruction & Compute Optimized)
* **Strategy**: Implements **Thread Coarsening** where each thread processes **two** output elements simultaneously.
* **Benefit**: 
    * **Data Reuse**: The middle numerical flux $f_{i+1/2}$ is computed once and reused for both $u_i$ and $u_{i+1}$ updates.
    * **ILP (Instruction Level Parallelism)**: Increases the arithmetic intensity per thread, allowing the GPU scheduler to hide instruction latency more effectively.

## 3. Benchmark Results
* **Hardware**: NVIDIA GeForce RTX 4050 (Laptop)
* **Problem Size**: $N = 2048$ grid points
* **Iterations**: $1000$ RK3 steps
* **CFL Number**: $0.4$

| Version | Total Execution Time (ms) | Avg Time per Step (ms) | Speedup (vs. Naive) |
| :--- | :--- | :--- | :--- |
| **Naive Baseline** | 40.850 | 0.041 | 1.00x |
| **Warp Shuffle** | 23.380 | 0.023 | 1.75x |
| **Shuffle + Register Tiling** | 21.987 | 0.022 | **1.86x** |

> *Note: Timings were measured using CUDA Events and include a warm-up phase to ensure the GPU is at peak clock frequency.*

## 4. Conclusion
The implementation demonstrates that for compute-intensive stencils like WENO5, performance is governed by the balance between memory bandwidth and arithmetic intensity. By shifting from global memory loads to **Warp-level communication** and **Register Tiling**, we achieved a near **2x speedup**, proving that fine-grained hardware-aware optimizations are critical for scientific computing on modern GPU architectures.
