#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ========================================================
// 1. GPU 工具函数与算子
// ========================================================

__device__ __inline__ float sqr(float x) { return x * x; }

// WENO5 核心重构算子：给定 5 个点，返回界面重构值 f_{i+1/2}
__device__ __inline__ float weno5_reconstruct(float v0, float v1, float v2, float v3, float v4) {
    float q0 = (1.0f/6.0f) * ( 2.0f*v0 - 7.0f*v1 + 11.0f*v2);
    float q1 = (1.0f/6.0f) * (-1.0f*v1 + 5.0f*v2 + 2.0f*v3);
    float q2 = (1.0f/6.0f) * ( 2.0f*v2 + 5.0f*v3 - 1.0f*v4);

    float b0 = (13.0f/12.0f) * sqr(v0 - 2.0f*v1 + v2) + 0.25f * sqr(v0 - 4.0f*v1 + 3.0f*v2);
    float b1 = (13.0f/12.0f) * sqr(v1 - 2.0f*v2 + v3) + 0.25f * sqr(v1 - v3);
    float b2 = (13.0f/12.0f) * sqr(v2 - 2.0f*v3 + v4) + 0.25f * sqr(3.0f*v2 - 4.0f*v3 + v4);

    const float eps = 1e-6f;
    float a0 = 0.1f / sqr(eps + b0);
    float a1 = 0.6f / sqr(eps + b1);
    float a2 = 0.3f / sqr(eps + b2);
    float asum = a0 + a1 + a2;

    return (a0 * q0 + a1 * q1 + a2 * q2) / asum;
}

__global__ void apply_bc_kernel(float* u, int N) {
    int i = threadIdx.x;
    if (i < 3) {
        u[i] = 1.0f;           // 流入边界
        u[N - 1 - i] = 0.125f; // 流出边界
    }
}

// 完整的右端项算子 L(u) = -(a/dx) * (f_right - f_left)
__device__ __inline__ float compute_L_naive(const float* u, int idx, float a_inv_dx) {
    float f_m3 = u[idx-3];
    float f_m2 = u[idx-2];
    float f_m1 = u[idx-1];
    float f_i  = u[idx];
    float f_p1 = u[idx+1];
    float f_p2 = u[idx+2];

    float flux_right = weno5_reconstruct(f_m2, f_m1, f_i, f_p1, f_p2);
    float flux_left  = weno5_reconstruct(f_m3, f_m2, f_m1, f_i, f_p1);

    return -a_inv_dx * (flux_right - flux_left);
}
/*
__device__ __inline__ float compute_L(const float* u, int idx, float a_inv_dx) {
    float f_i  = u[idx];
    int lane_id = idx % 32;
    float f_m3 = (lane_id - 3 >= 0) ? __shfl_up_sync(0xffffffff, f_i, 3): u[idx - 3];
    float f_m2 = (lane_id - 2 >= 0) ? __shfl_up_sync(0xffffffff, f_i, 2): u[idx - 2];
    float f_m1 = (lane_id - 1 >= 0) ? __shfl_up_sync(0xffffffff, f_i, 1): u[idx - 1];
    float f_p1 = (lane_id + 1 <= 31) ? __shfl_down_sync(0xffffffff, f_i, 1): u[idx + 1];
    float f_p2 = (lane_id + 2 <= 31) ? __shfl_down_sync(0xffffffff, f_i, 2): u[idx + 2];

    float flux_right = weno5_reconstruct(f_m2, f_m1, f_i, f_p1, f_p2);
    float flux_left  = weno5_reconstruct(f_m3, f_m2, f_m1, f_i, f_p1);

    return -a_inv_dx * (flux_right - flux_left);
}
*/

//warp_shuffle version
__device__ __inline__ float compute_L_warp_shuffle(const float* u, int idx, float a_inv_dx) {
    int lane_id = threadIdx.x & 0x1f; // 使用位运算取模，比 % 32 快
    float f_i = u[idx];

    // 1. 全员洗牌：不管边界，大家一起 Shuffle
    // __shfl_up/down 在越界时（src_lane < 0 或 > 31）会返回调用者自己的原值
    float f_m1 = __shfl_up_sync(0xffffffff, f_i, 1);
    float f_m2 = __shfl_up_sync(0xffffffff, f_i, 2);
    float f_m3 = __shfl_up_sync(0xffffffff, f_i, 3);
    float f_p1 = __shfl_down_sync(0xffffffff, f_i, 1);
    float f_p2 = __shfl_down_sync(0xffffffff, f_i, 2);

    // 2. 边界补位：只有处于边缘的线程去读显存
    // 这里虽然有 if，但由于只有极少数线程执行，分支发散的影响很小
    if (lane_id < 1) f_m1 = u[idx - 1];
    if (lane_id < 2) f_m2 = u[idx - 2];
    if (lane_id < 3) f_m3 = u[idx - 3];
    if (lane_id > 30) f_p1 = u[idx + 1];
    if (lane_id > 29) f_p2 = u[idx + 2];

    // 3. 正常的 WENO 重构
    float flux_right = weno5_reconstruct(f_m2, f_m1, f_i, f_p1, f_p2);
    float flux_left  = weno5_reconstruct(f_m3, f_m2, f_m1, f_i, f_p1);

    return -a_inv_dx * (flux_right - flux_left);
}


//warp shuffle register tiled version(best)
struct L_pair { float left; float right; };
__device__ __inline__ L_pair compute_L_warp_shuffle_tiled(const float* u, int idx, float a_inv_dx) {
    int lane_id = threadIdx.x & 0x1f; // 使用位运算取模，比 % 32 快
    float f_i = u[idx];

    // 1. 全员洗牌：不管边界，大家一起 Shuffle
    // __shfl_up/down 在越界时（src_lane < 0 或 > 31）会返回调用者自己的原值
    float f_m1 = __shfl_up_sync(0xffffffff, f_i, 1);
    float f_m2 = __shfl_up_sync(0xffffffff, f_i, 2);
    float f_m3 = __shfl_up_sync(0xffffffff, f_i, 3);
    float f_p1 = __shfl_down_sync(0xffffffff, f_i, 1);
    float f_p2 = __shfl_down_sync(0xffffffff, f_i, 2);
    float f_p3 = __shfl_down_sync(0xffffffff, f_i, 3);

    // 2. 边界补位：只有处于边缘的线程去读显存
    // 这里虽然有 if，但由于只有极少数线程执行，分支发散的影响很小
    if (lane_id < 1) f_m1 = u[idx - 1];
    if (lane_id < 2) f_m2 = u[idx - 2];
    if (lane_id < 3) f_m3 = u[idx - 3];
    if (lane_id > 30) f_p1 = u[idx + 1];
    if (lane_id > 29) f_p2 = u[idx + 2];
    if (lane_id > 28) f_p3 = u[idx + 3];

    // 3. 正常的 WENO 重构
    float flux_right = weno5_reconstruct(f_m1, f_i, f_p1, f_p2, f_p3);
    float flux_mid = weno5_reconstruct(f_m2, f_m1, f_i, f_p1, f_p2);
    float flux_left  = weno5_reconstruct(f_m3, f_m2, f_m1, f_i, f_p1);
    L_pair res;
    res.left = -a_inv_dx * (flux_right - flux_mid);
    res.right = -a_inv_dx * (flux_mid - flux_left);
    return res;
}

// ========================================================
// 2. TVD-RK3 三个子步的 Kernels
// ========================================================
enum Method { NAIVE, SHUFFLE, TILED };
// Stage 1: u(1) = u^n + dt * L(u^n)
template<Method M>
__global__ void kernel_rk_stage1(const float* u_n, float* u_1, int N, float dt, float a_inv_dx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (M == TILED) {
        int idx = tid * 2; // 线程处理物理索引 2i 和 2i+1
        if (idx >= 3 && idx < N - 4) {
            L_pair Ls = compute_L_warp_shuffle_tiled(u_n, idx, a_inv_dx);
            u_1[idx]   = u_n[idx]   + dt * Ls.left;
            u_1[idx+1] = u_n[idx+1] + dt * Ls.right;
        }
    } else {
        if (tid >= 3 && tid < N - 2) {
            float L = (M == NAIVE) ? compute_L_naive(u_n, tid, a_inv_dx) : compute_L_warp_shuffle(u_n, tid, a_inv_dx);
            u_1[tid] = u_n[tid] + dt * L;
        }
    }
}

// Stage 2: u(2) = 3/4 * u^n + 1/4 * u(1) + 1/4 * dt * L(u(1))
template<Method M>
__global__ void kernel_rk_stage2(const float* u_n, const float* u_1, float* u_2, int N, float dt, float a_inv_dx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (M == TILED) {
        int idx = tid * 2;
        if (idx >= 3 && idx < N - 4) {
            L_pair Ls = compute_L_warp_shuffle_tiled(u_1, idx, a_inv_dx);
            u_2[idx]   = 0.75f * u_n[idx]   + 0.25f * u_1[idx]   + 0.25f * dt * Ls.left;
            u_2[idx+1] = 0.75f * u_n[idx+1] + 0.25f * u_1[idx+1] + 0.25f * dt * Ls.right;
        }
    } else {
        if (tid >= 3 && tid < N - 2) {
            float L = (M == NAIVE) ? compute_L_naive(u_1, tid, a_inv_dx) : compute_L_warp_shuffle(u_1, tid, a_inv_dx);
            u_2[tid] = 0.75f * u_n[tid] + 0.25f * u_1[tid] + 0.25f * dt * L;
        }
    }
}

// Stage 3: u^(n+1) = 1/3 * u^n + 2/3 * u(2) + 2/3 * dt * L(u(2))
template<Method M>
__global__ void kernel_rk_stage3(const float* u_n, const float* u_2, float* u_next, int N, float dt, float a_inv_dx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (M == TILED) {
        int idx = tid * 2;
        if (idx >= 3 && idx < N - 4) {
            L_pair Ls = compute_L_warp_shuffle_tiled(u_2, idx, a_inv_dx);
            u_next[idx]   = (1.0f/3.0f) * u_n[idx]   + (2.0f/3.0f) * u_2[idx]   + (2.0f/3.0f) * dt * Ls.left;
            u_next[idx+1] = (1.0f/3.0f) * u_n[idx+1] + (2.0f/3.0f) * u_2[idx+1] + (2.0f/3.0f) * dt * Ls.right;
        }
    } else {
        if (tid >= 3 && tid < N - 2) {
            float L = (M == NAIVE) ? compute_L_naive(u_2, tid, a_inv_dx) : compute_L_warp_shuffle(u_2, tid, a_inv_dx);
            u_next[tid] = (1.0f/3.0f) * u_n[tid] + (2.0f/3.0f) * u_2[tid] + (2.0f/3.0f) * dt * L;
        }
    }
}
// ========================================================
// 3. Host 主程序
// ========================================================

int main() {
    const int N = 4096; // 增加规模以观察 Tiling 优势
    const float dx = 1.0f / N;
    const float dt = 0.4f * dx / 1.0f;
    const float a_inv_dx = 1.0f / dx;
    const int max_steps = 1000;

    float *d_u_n, *d_u_1, *d_u_2;
    cudaMalloc(&d_u_n, N * sizeof(float));
    cudaMalloc(&d_u_1, N * sizeof(float));
    cudaMalloc(&d_u_2, N * sizeof(float));

    std::vector<float> h_init(N);
    for (int i = 0; i < N; i++) h_init[i] = (i * dx < 0.5f) ? 1.0f : 0.125f;

    auto benchmark = [&](Method m, const char* name) {
        cudaMemcpy(d_u_n, h_init.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        
        // 执行配置：Tiled 模式下线程数减半
        int threads = 256;
        int blocks = (m == TILED) ? ((N / 2) + threads - 1) / threads : (N + threads - 1) / threads;

        cudaEventRecord(start);
        for (int step = 0; step < max_steps; step++) {
            apply_bc_kernel<<<1, 32>>>(d_u_n, N);
            if (m == NAIVE) {
                kernel_rk_stage1<NAIVE><<<blocks, threads>>>(d_u_n, d_u_1, N, dt, a_inv_dx);
                kernel_rk_stage2<NAIVE><<<blocks, threads>>>(d_u_n, d_u_1, d_u_2, N, dt, a_inv_dx);
                kernel_rk_stage3<NAIVE><<<blocks, threads>>>(d_u_n, d_u_2, d_u_n, N, dt, a_inv_dx);
            } else if (m == SHUFFLE) {
                kernel_rk_stage1<SHUFFLE><<<blocks, threads>>>(d_u_n, d_u_1, N, dt, a_inv_dx);
                kernel_rk_stage2<SHUFFLE><<<blocks, threads>>>(d_u_n, d_u_1, d_u_2, N, dt, a_inv_dx);
                kernel_rk_stage3<SHUFFLE><<<blocks, threads>>>(d_u_n, d_u_2, d_u_n, N, dt, a_inv_dx);
            } else {
                kernel_rk_stage1<TILED><<<blocks, threads>>>(d_u_n, d_u_1, N, dt, a_inv_dx);
                kernel_rk_stage2<TILED><<<blocks, threads>>>(d_u_n, d_u_1, d_u_2, N, dt, a_inv_dx);
                kernel_rk_stage3<TILED><<<blocks, threads>>>(d_u_n, d_u_2, d_u_n, N, dt, a_inv_dx);
            }
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("[%s] Total Time: %.3f ms (Avg: %.3f ms/step)\n", name, ms, ms/max_steps);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    };

    benchmark(NAIVE, "Naive");
    benchmark(SHUFFLE, "Warp Shuffle");
    benchmark(TILED, "Shuffle + Register Tiling");

    // 保存最后一次结果 (Shuffle 版本)
    std::vector<float> h_res(N);
    cudaMemcpy(h_res.data(), d_u_n, N * sizeof(float), cudaMemcpyDeviceToHost);
    FILE *fp = fopen("benchmark_result.csv", "w");
    fprintf(fp, "x,u\n");
    for (int i = 0; i < N; i++) fprintf(fp, "%f,%f\n", i * dx, h_res[i]);
    fclose(fp);

    cudaFree(d_u_n); cudaFree(d_u_1); cudaFree(d_u_2);
    return 0;
}
