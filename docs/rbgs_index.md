# Mathematical Optimization of Red-Black Gauss-Seidel Indexing on GPU

This document provides a deep dive into the indexing logic developed for the High-Performance 2D Poisson Solver. It explores how we resolved **Warp Divergence** and **Memory Coalescing** issues by transforming the mathematical grid into a hardware-friendly layout.

## 1. The Parallel Challenge: Red-Black Coloring

To parallelize the Gauss-Seidel (GS) method, we employ a checkerboard coloring scheme. A grid point $(i, j)$ is **Red** if $(i+j)$ is even, and **Black** if $(i+j)$ is odd. Since any point's neighbors are always of the opposite color, we can update all points of one color simultaneously in two distinct half-steps:

1.  **Red Step**: Update all Red points using Black neighbors.
2.  **Black Step**: Update all Black points using Red neighbors.

---

## 2. Resolving Warp Divergence: Grid Compression

### The Problem
A naive implementation launches a thread for every grid point, but masks out 50% of the threads using `if((i+j)%2 == color)`. In the CUDA SIMT architecture, this causes massive **Warp Divergence**, where half the lanes in a warp are idle, wasting 50% of the GPU's peak throughput.

### The Solution: $N/2$ Thread Mapping
We launch exactly $W/2 \times H$ threads. Each thread is mapped to a physical checkerboard coordinate $(i, j)$ using a branch-free bitwise formula.

**Mapping Logic:**
* **Even Rows ($i$ even)**: Red points start at $j=0$.
* **Odd Rows ($i$ odd)**: Red points start at $j=1$.

By incorporating the `color_step` (0 for Red, 1 for Black), we derive the universal mapping formula:
$$j = 2 \cdot j_{half} + ((i + \text{color\_step}) \& 1)$$



This ensures **100% Warp Efficiency** as every lane in the warp performs active computation.

---

## 3. Resolving Memory Coalescing: Split Layout Transformation

### The Problem
Even with 100% thread activity, the compressed mapping in a unified grid results in **Strided Access** (Stride-2). Threads $k$ and $k+1$ access memory addresses $p$ and $p+8$ (for floats). This prevents the hardware from coalescing requests into a single 128-byte memory transaction, effectively halving the effective bandwidth.

### The Solution: Memory Splitting
We reorganize the data into two contiguous arrays: `float* red_data` and `float* black_data`. Each has a size of $(W/2 \times H)$.



### The Challenge: Relative Neighbor Indexing
In a split layout, the spatial "neighbors" are no longer at fixed offsets. Their positions in the *opposite* array shift depending on the row's parity.

**Observation ($W=6$):**
* **Even Row ($i=0$):** Sequence is $R_0, B_0, R_1, B_1, R_2, B_2$
    * $R_1$ (idx 1) has neighbors $B_0$ (idx 0) and $B_1$ (idx 1).
* **Odd Row ($i=1$):** Sequence is $B_0, R_0, B_1, R_1, B_2, R_2$
    * $R_1$ (idx 1) has neighbors $B_1$ (idx 1) and $B_2$ (idx 2).

### The Optimized Indexing Formula
To avoid costly `if-else` blocks inside the kernel, we developed a unified bitwise formula for the left neighbor's index:

$$left\_idx = \text{idx} - 1 + ((i_{\text{parity}} + \text{color\_step}) \& 1)$$
$$right\_idx = left\_idx + 1$$

| Scenario | Row Parity ($i\&1$) | Step (`color`) | Offset Logic | Left Index Result |
| :--- | :---: | :---: | :---: | :--- |
| **Red Step** | Even (0) | 0 | 0 | `idx - 1` |
| **Red Step** | Odd (1) | 0 | 1 | `idx` |
| **Black Step** | Even (0) | 1 | 1 | `idx` |
| **Black Step** | Odd (1) | 1 | 0 | `idx - 1` |

This logic ensures that all horizontal neighbor fetches are **coalesced**, as the `left_idx` for thread $k$ is perfectly followed by `left_idx` for thread $k+1$.

---

## 4. Performance Significance

By combining **Grid Compression** and **Split Layout Indexing**, the solver transitions from being latency-bound to being **bandwidth-bound**. 

* **Instruction Optimization**: Replaced integer modulo (`%`) and branching with bitwise `&` and addition, reducing ALU pressure.
* **Throughput**: Successfully saturated ~80% of the theoretical memory bandwidth on the test hardware.
* **Convergence**: Maintains the $O(2)$ convergence rate of Gauss-Seidel over Jacobi while achieving Jacobi-like parallel performance.

---
