# Performance Sensitivity Analysis of 1D Lax-Wendroff Stencil

## 1. Experiment Overview
This experiment evaluates the performance of the **Lax-Wendroff** scheme for the 1D linear advection equation across four distinct CUDA implementation strategies:
* **Naive (Global Memory)**: Direct global memory access with standard indexing.
* **Shared Memory**: Explicit management of Halo cells (Ghost cells) with thread synchronization.
* **Warp Shuffle**: Utilizing hardware primitives for register-level data exchange within warps.
* **Register Tiling (2x ILP)**: Increasing the workload per thread to improve Instruction-Level Parallelism and reduce memory instruction overhead.

The grid size $N$ ranges from $2^{10}$ to $2^{24}$ to assess performance under different memory hierarchy pressures, measured in **GUpdates/s** (Giga-Updates per second).

---

## 2. Experimental Results

| $N$ | Naive | Shared | Shuffle | Tiling (2x) | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $2^{10}$ | 0.150 | 0.174 | 0.187 | 0.172 | Shuffle |
| $2^{16}$ | 11.11 | 9.95 | 11.55 | 10.28 | Shuffle |
| $2^{20}$ | 68.31 | 60.84 | 60.23 | **109.75** | **Tiling (2x)** |
| $2^{22}$ | 22.86 | 22.82 | 22.84 | **33.43** | **Tiling (2x)** |
| $2^{24}$ | 21.77 | 22.18 | 21.91 | 21.37 | Shared/Naive |

---

## 3. Deep Performance Analysis

### 3.1 Why Shared Memory and Shuffle Underperformed Naive
In the 1D stencil computation, the Naive version proved remarkably resilient. This is attributed to:
* **Hardware Prefetching & L1/L2 Cache Efficiency**: Since 1D access is strictly contiguous (Stride=1), modern GPU L1/L2 caches are highly efficient at spatial locality optimization. The "manual" movement to Shared Memory is essentially redundant as hardware prefetching accomplishes the same goal without overhead.
* **Synchronization Overhead**: The Shared Memory approach introduces `__syncthreads()`, which forces warp stalls. Additionally, manual index calculations for shared arrays increase integer arithmetic overhead.
* **NVCC Compiler Intelligence**: In the case of Warp Shuffle, the compiler (NVCC) often automatically optimizes adjacent register access for simple stencil patterns, diminishing the marginal returns of manual shuffle primitives.



### 3.2 The Register Tiling (2x ILP) Breakthrough
The Tiling scheme achieved a dominant peak of **109.75 GU/s** near $N=2^{20}$. This validates several key HPC optimization principles:
* **Reduction in Load Instructions**: Calculating 2 output points requires only 4 Load instructions instead of 6 in the standard mode. This 33% reduction in memory instructions significantly lowers the pressure on the Instruction Issue unit.
* **Instruction-Level Parallelism (ILP)**: Processing multiple points per thread increases the number of independent floating-point operations (FMA chains). This allows the GPU scheduler to better hide global memory access latency by overlapping computation with pending memory fetches.
* **Register Residency**: Intermediate grid data is reused directly within registers, bypassing the lookup latency of the cache hierarchy entirely.



### 3.3 The DRAM Bandwidth Wall
Beyond $N \ge 2^{22}$ (data footprint > 16MB), the workload exceeds the L2 Cache capacity. At this scale, the bottleneck shifts entirely to **Physical VRAM (DRAM) Bandwidth**. Since the total amount of data moved remains constant across all schemes, performance converges to approximately 21~22 GU/s, representing the hardware's theoretical limit for global memory writes.

---

## 4. Conclusion & Outlook
1.  **1D Stencil Insights**: For contiguous 1D workloads, manual Shared Memory optimization provides limited gains due to the efficiency of the L2 cache. **Register Tiling** is the most effective strategy to break through cache bandwidth limits by reducing instruction count and increasing ILP.
2.  **Transition to 2D**: In a 2D domain, "vertical" access (Stride >> 1) will trigger massive Cache Misses. In that scenario, Shared Memory Tiling will become the critical factor in transforming non-coalesced access into high-performance coalesced reads.
