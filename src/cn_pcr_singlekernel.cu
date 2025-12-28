#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ========================================================
// 宏定义与配置
// ========================================================
#define N 1024               // 求解规模，刚好占满一个 Block
#define BLOCK_SIZE 1024      // 线程块大小
#define BENCH_STEPS 100      // 性能测试迭代步数

// ========================================================
// 1. GPU Kernels (极致融合版)
// ========================================================

// 1.1 GPU 侧初始化：直接生成 CN 矩阵系数，消除数据传输
__global__ void init_coeffs_kernel(float* a, float* b, float* c, float r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = (i == 0) ? 0.0f : -r;     // 物理左边界
    b[i] = 1.0f + 2.0f * r;
    c[i] = (i == n - 1) ? 0.0f : -r; // 物理右边界
}

// 1.2 计算右端项 d
__global__ void compute_rhs_kernel(const float* u, float* d, int n, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float u_l = (i > 0) ? u[i - 1] : 0.0f; 
    float u_c = u[i];
    float u_r = (i < n - 1) ? u[i + 1] : 0.0f;
    d[i] = r * u_l + (1.0f - 2.0f * r) * u_c + r * u_r;
}

// 1.3 终极融合 PCR 算子
// MODE 0: 仅使用 Shared Memory
// MODE 1: 混合优化 (Warp Shuffle + Shared Memory)
template<int MODE>
__global__ void pcr_fused_solver_kernel(
    const float* a_in, const float* b_in, const float* c_in, const float* d_in,
    float* u_out, int n) 
{
    __shared__ float s_a[BLOCK_SIZE];
    __shared__ float s_b[BLOCK_SIZE];
    __shared__ float s_c[BLOCK_SIZE];
    __shared__ float s_d[BLOCK_SIZE];

    int tid = threadIdx.x;
    int lane = tid & 0x1f;
    if (tid >= n) return;

    // 步骤 A: 一次性载入 (Coalesced Load)
    s_a[tid] = a_in[tid];
    s_b[tid] = b_in[tid];
    s_c[tid] = c_in[tid];
    s_d[tid] = d_in[tid];
    __syncthreads();

    // 步骤 B: 内部循环迭代，无需返回显存
    for (int stride = 1; stride < n; stride <<= 1) {
        float ai = s_a[tid], bi = s_b[tid], ci = s_c[tid], di = s_d[tid];
        float al = 0, bl = 1, cl = 0, dl = 0, ar = 0, br = 1, cr = 0, dr = 0;

        if (MODE == 1 && stride < 32) {
            // 模式 1：利用 Warp Shuffle 减少访存
            // 注意：Shuffle 必须在 if 外部进行全 Warp 同步以防死锁
            float bl_s = __shfl_up_sync(0xffffffff, bi, stride);
            float al_s = __shfl_up_sync(0xffffffff, ai, stride);
            float cl_s = __shfl_up_sync(0xffffffff, ci, stride);
            float dl_s = __shfl_up_sync(0xffffffff, di, stride);
            float br_s = __shfl_down_sync(0xffffffff, bi, stride);
            float ar_s = __shfl_down_sync(0xffffffff, ai, stride);
            float cr_s = __shfl_down_sync(0xffffffff, ci, stride);
            float dr_s = __shfl_down_sync(0xffffffff, di, stride);

            if (lane >= stride) { bl = bl_s; al = al_s; cl = cl_s; dl = dl_s; }
            if (lane + stride < 32) { br = br_s; ar = ar_s; cr = cr_s; dr = dr_s; }
        } else {
            // 模式 0：标准的 Shared Memory 访问
            if (tid - stride >= 0) {
                al = s_a[tid - stride]; bl = s_b[tid - stride];
                cl = s_c[tid - stride]; dl = s_d[tid - stride];
            }
            if (tid + stride < n) {
                ar = s_a[tid + stride]; br = s_b[tid + stride];
                cr = s_c[tid + stride]; dr = s_d[tid + stride];
            }
        }

        float alpha = ai / bl;
        float gamma = ci / br;

        __syncthreads(); // 必须同步，防止读取到这一轮更新后的值
        s_a[tid] = -alpha * al;
        s_c[tid] = -gamma * cr;
        s_b[tid] = bi - alpha * cl - gamma * ar;
        s_d[tid] = di - alpha * dl - gamma * dr;
        __syncthreads(); // 确保写回完成，进入下一轮迭代
    }

    // 步骤 C: 计算最终结果并写回显存
    u_out[tid] = s_d[tid] / s_b[tid];
}

// ========================================================
// 2. Host Logic & Benchmark
// ========================================================

void solve_and_benchmark() {
    const float L = 1.0f, alpha_c = 0.01f, dt = 0.0001f;
    const float dx = L / (N - 1), r = (alpha_c * dt) / (2.0f * dx * dx);
    
    std::vector<float> h_u(N);
    for (int i = 0; i < N; i++) h_u[i] = (i * dx > 0.4f && i * dx < 0.6f) ? 1.0f : 0.0f;

    float *d_u, *d_a, *d_b, *d_c, *d_d;
    cudaMalloc(&d_u, N * sizeof(float));
    cudaMalloc(&d_a, N * sizeof(float)); cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float)); cudaMalloc(&d_d, N * sizeof(float));

    cudaMemcpy(d_u, h_u.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    std::cout << "--- 1D Heat Equation PCR Fused Solver Benchmark ---" << std::endl;
    std::cout << "Grid Size: " << N << " | r: " << r << std::endl;

    // 测试模式 0 (Naive Shared)
    float time_m0 = 0;
    for (int s = 0; s < BENCH_STEPS; s++) {
        compute_rhs_kernel<<<1, BLOCK_SIZE>>>(d_u, d_d, N, r);
        init_coeffs_kernel<<<1, BLOCK_SIZE>>>(d_a, d_b, d_c, r, N);
        cudaEventRecord(start);
        pcr_fused_solver_kernel<0><<<1, BLOCK_SIZE>>>(d_a, d_b, d_c, d_d, d_u, N); // 直接写回 d_u
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        time_m0 += ms;
    }
    std::cout << "Mode 0 (Naive Shared) Avg Time: " << (time_m0 / BENCH_STEPS) << " ms" << std::endl;

    // 测试模式 1 (Hybrid Shuffle+Shared)
    float time_m1 = 0;
    for (int s = 0; s < BENCH_STEPS; s++) {
        compute_rhs_kernel<<<1, BLOCK_SIZE>>>(d_u, d_d, N, r);
        init_coeffs_kernel<<<1, BLOCK_SIZE>>>(d_a, d_b, d_c, r, N);
        cudaEventRecord(start);
        pcr_fused_solver_kernel<1><<<1, BLOCK_SIZE>>>(d_a, d_b, d_c, d_d, d_u, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        time_m1 += ms;
    }
    std::cout << "Mode 1 (Hybrid Shuffle) Avg Time: " << (time_m1 / BENCH_STEPS) << " ms" << std::endl;

    // 结果导出与清理
    cudaMemcpy(h_u.data(), d_u, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_u); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    solve_and_benchmark();
    return 0;
}
