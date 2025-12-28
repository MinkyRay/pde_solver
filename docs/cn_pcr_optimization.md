# 1D Parabolic PDE Solver: High-Performance GPU Optimization of Crank-Nicolson via PCR

## Project Overview
This project implements a high-performance 1D Heat Equation solver ($u_t = \alpha u_{xx}$) using the **Crank-Nicolson (CN)** implicit scheme on NVIDIA GPUs. To solve the resulting tridiagonal system $Ax=b$ efficiently, I utilized the **Parallel Cyclic Reduction (PCR)** algorithm. The project explores the transition from naive implementations to extreme optimizations using **Kernel Fusion**, **Shared Memory**, and **Warp Shuffle** primitives.

---

## Optimization Journey

### 1. Multi-Kernel Implementations: The Overhead Bottleneck
Initially, I compared a **Naive PCR** version with a **Warp Shuffle** optimized version, where each iteration (doubling the stride) was launched as a separate kernel.
* **Result**: At a scale of $N=1024$, the performance of both versions was nearly identical.
* **Analysis**: The execution time was dominated by **Kernel Launch Overhead** (typically 10-30 $\mu s$ on Windows WDDM) rather than the arithmetic complexity. 
* **Hardware Influence**: For small $N$, the data residency in the L2 cache masked global memory access latencies, making the register-level benefits of shuffle less apparent.

### 2. Single-Kernel Fusion: Shared Memory & Intra-Kernel Loops
To eliminate launch overhead, I fused all $\log_2 N$ iterations into a **single CUDA kernel**.
* **Strategy**: Data is loaded into **Shared Memory** once, and all PCR steps are performed in an internal loop with `__syncthreads()` barriers.
* **Outcome**: This approach reduced global memory transactions by nearly $16\times$ and achieved a massive speedup (estimated 10x-50x) compared to the multi-kernel approach by eliminating redundant round-trips to Global Memory.

### 3. Hybrid Mode: Warp Shuffle + Shared Memory
I implemented a hybrid solver that utilizes **Warp Shuffle** for intra-warp stages ($stride < 32$) and **Shared Memory** for inter-warp stages.
* **Observation**: Performance was slightly lower than pure Shared Memory for $N=1024$.
* **Trade-offs**: While Warp Shuffle provides lower latency than Shared Memory, the hybrid logic introduces higher **Register Pressure** and additional instruction overhead to ensure synchronization safety.
* **Scalability**: This hybrid approach is designed to scale for larger systems where memory bandwidth becomes the primary constraint and $N$ exceeds the capacity of a single block's shared memory.

---

## Technical Challenge: Warp Shuffle Deadlocks
A key technical challenge involved resolving deadlocks in the Warp Shuffle implementation on modern NVIDIA architectures (Volta/Ada Lovelace).
* **The Issue**: Using `__shfl_up_sync` with a full mask (`0xffffffff`) inside a divergent `if` block caused threads to hang, as the mask requires all 32 threads in a warp to converge at the instruction.
* **The Solution**: I decoupled the synchronization primitive from the divergent control flow by performing a full-warp shuffle followed by conditional usage of the results, ensuring warp-level consistency.

---

## Conclusion & Future Work
* **Hardware-Algorithm Synergy**: Efficient HPC requires mapping mathematical algorithms to the specific hierarchical memory structure of the GPU.
* **Academic Application**: This work serves as a foundational component for my planned **2D ADI (Alternating Direction Implicit)** solver, which will utilize this 1D PCR solver as its core row/column solver.
* **Portfolio Impact**: This project demonstrates proficiency in CUDA C++, numerical analysis, and low-level hardware optimization, supporting my future PhD applications in Canada or Europe.

---
*Developed by a Master's student in Computational Mathematics at Nankai University, with research interests in GPU acceleration and LLM compression.*
