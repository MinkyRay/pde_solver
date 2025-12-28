#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ========================================================
// 宏定义与配置
// ========================================================
#define N 1024
#define USE_SHUFFLE 1  // 1: 使用 Warp Shuffle, 0: 使用 Naive PCR

// ========================================================
// 1. GPU Kernels
// ========================================================

// 1.1 初始化矩阵系数 (直接在 GPU 生成，避免 Host-Device 拷贝)
__global__ void init_coeffs_kernel(float* a, float* b, float* c, float r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // CN 格式：b = 1+2r, a = -r, c = -r
    // 物理边界处理：第一行无左项，最后一行无右项
    a[i] = (i == 0) ? 0.0f : -r;
    b[i] = 1.0f + 2.0f * r;
    c[i] = (i == n - 1) ? 0.0f : -r;
}

// 1.2 计算右端项 d (显式部分)
__global__ void compute_rhs_kernel(const float* u, float* d, int n, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float u_l = (i > 0) ? u[i - 1] : 0.0f; 
    float u_c = u[i];
    float u_r = (i < n - 1) ? u[i + 1] : 0.0f;

    d[i] = r * u_l + (1.0f - 2.0f * r) * u_c + r * u_r;
}

// 1.3 Naive PCR 算法单步迭代
__global__ void pcr_step_kernel(
    const float* a_in, const float* b_in, const float* c_in, const float* d_in,
    float* a_out, float* b_out, float* c_out, float* d_out,
    int n, int stride) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ai = a_in[i], bi = b_in[i], ci = c_in[i], di = d_in[i];
    float al = 0, cl = 0, dl = 0, ar = 0, cr = 0, dr = 0;
    float alpha = 0, gamma = 0;

    if (i - stride >= 0) {
        float bl = b_in[i - stride];
        alpha = ai / bl;
        al = a_in[i - stride];
        cl = c_in[i - stride];
        dl = d_in[i - stride];
    }
    if (i + stride < n) {
        float br = b_in[i + stride];
        gamma = ci / br;
        ar = a_in[i + stride];
        cr = c_in[i + stride];
        dr = d_in[i + stride];
    }

    a_out[i] = -alpha * al;
    c_out[i] = -gamma * cr;
    b_out[i] = bi - alpha * cl - gamma * ar;
    d_out[i] = di - alpha * dl - gamma * dr;
}

// 1.4 Warp Shuffle 优化的 PCR 迭代
/*
__global__ void pcr_step_kernel_shuffle(
    const float* a_in, const float* b_in, const float* c_in, const float* d_in,
    float* a_out, float* b_out, float* c_out, float* d_out,
    int n, int stride) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int lane = threadIdx.x & 0x1f;

    float ai = a_in[i], bi = b_in[i], ci = c_in[i], di = d_in[i];
    float al = 0, bl = 1, cl = 0, dl = 0, ar = 0, br = 1, cr = 0, dr = 0;

    // 处理左邻居
    if (i - stride >= 0) {
        if (stride < 32 && lane >= stride) {
            bl = __shfl_up_sync(0xffffffff, bi, stride);
            al = __shfl_up_sync(0xffffffff, ai, stride);
            cl = __shfl_up_sync(0xffffffff, ci, stride);
            dl = __shfl_up_sync(0xffffffff, di, stride);
        } else {
            bl = b_in[i - stride];
            al = a_in[i - stride];
            cl = c_in[i - stride];
            dl = d_in[i - stride];
        }
    }
    // 处理右邻居
    if (i + stride < n) {
        if (stride < 32 && lane + stride < 32) {
            br = __shfl_down_sync(0xffffffff, bi, stride);
            ar = __shfl_down_sync(0xffffffff, ai, stride);
            cr = __shfl_down_sync(0xffffffff, ci, stride);
            dr = __shfl_down_sync(0xffffffff, di, stride);
        } else {
            br = b_in[i + stride];
            ar = a_in[i + stride];
            cr = c_in[i + stride];
            dr = d_in[i + stride];
        }
    }

    float alpha = ai / bl;
    float gamma = ci / br;
    a_out[i] = -alpha * al;
    c_out[i] = -gamma * cr;
    b_out[i] = bi - alpha * cl - gamma * ar;
    d_out[i] = di - alpha * dl - gamma * dr;
}
*/

__global__ void pcr_step_kernel_shuffle(
    const float* a_in, const float* b_in, const float* c_in, const float* d_in,
    float* a_out, float* b_out, float* c_out, float* d_out,
    int n, int stride) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 注意：如果 N 不是 32 的倍数，不满 32 人的最后那个 Warp 会在这里死锁
    // 对于 N=1024 没问题，但为了严谨，我们先标记活跃状态
    bool active = (i < n);

    float ai = active ? a_in[i] : 0.0f;
    float bi = active ? b_in[i] : 1.0f; // 避免除以 0
    float ci = active ? c_in[i] : 0.0f;
    float di = active ? d_in[i] : 0.0f;

    int lane = threadIdx.x & 0x1f;

    // --- 关键修正：全 Warp 范围内预先洗牌 ---
    // 无论是否符合 if 条件，所有 32 个线程都必须执行这两组指令
    float bl_s = __shfl_up_sync(0xffffffff, bi, stride);
    float al_s = __shfl_up_sync(0xffffffff, ai, stride);
    float cl_s = __shfl_up_sync(0xffffffff, ci, stride);
    float dl_s = __shfl_up_sync(0xffffffff, di, stride);

    float br_s = __shfl_down_sync(0xffffffff, bi, stride);
    float ar_s = __shfl_down_sync(0xffffffff, ai, stride);
    float cr_s = __shfl_down_sync(0xffffffff, ci, stride);
    float dr_s = __shfl_down_sync(0xffffffff, di, stride);

    if (!active) return; // 处理完同步后再让多余线程退出

    float al = 0, bl = 1, cl = 0, dl = 0;
    float ar = 0, br = 1, cr = 0, dr = 0;

    // 处理左邻居
    if (i - stride >= 0) {
        // 如果邻居在同一个 Warp 内，直接用预洗牌的结果
        if (stride < 32 && lane >= stride) {
            bl = bl_s; al = al_s; cl = cl_s; dl = dl_s;
        } else {
            bl = b_in[i - stride]; al = a_in[i - stride];
            cl = c_in[i - stride]; dl = d_in[i - stride];
        }
    }

    // 处理右邻居
    if (i + stride < n) {
        if (stride < 32 && lane + stride < 32) {
            br = br_s; ar = ar_s; cr = cr_s; dr = dr_s;
        } else {
            br = b_in[i + stride]; ar = ar_s; cr = cr_s; dr = dr_s;
        }
    }

    float alpha = ai / bl;
    float gamma = ci / br;
    a_out[i] = -alpha * al;
    c_out[i] = -gamma * cr;
    b_out[i] = bi - alpha * cl - gamma * ar;
    d_out[i] = di - alpha * dl - gamma * dr;
}

__global__ void finalize_kernel(float* u, const float* b, const float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) u[i] = d[i] / b[i];
}

// ========================================================
// 2. Host Solver
// ========================================================

void run_simulation() {
    const float L = 1.0f, alpha_c = 0.01f, dt = 0.0001f;
    const float dx = L / (N - 1);
    const float r = (alpha_c * dt) / (2.0f * dx * dx);
    const int steps = 100;

    std::vector<float> h_u(N);
    for (int i = 0; i < N; i++) {
        float x = i * dx;
        h_u[i] = (x > 0.4f && x < 0.6f) ? 1.0f : 0.0f;
    }

    float *d_u, *d_a[2], *d_b[2], *d_c[2], *d_d[2];
    cudaMalloc(&d_u, N * sizeof(float));
    for(int j=0; j<2; j++) {
        cudaMalloc(&d_a[j], N*sizeof(float)); cudaMalloc(&d_b[j], N*sizeof(float));
        cudaMalloc(&d_c[j], N*sizeof(float)); cudaMalloc(&d_d[j], N*sizeof(float));
    }

    cudaMemcpy(d_u, h_u.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // 计时器
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float total_solver_time = 0;

    std::cout << "Solver: " << (USE_SHUFFLE ? "Warp Shuffle" : "Naive") << " | r = " << r << std::endl;

    for (int s = 0; s < steps; s++) {
        compute_rhs_kernel<<<blocks, threads>>>(d_u, d_d[0], N, r);
        init_coeffs_kernel<<<blocks, threads>>>(d_a[0], d_b[0], d_c[0], r, N);

        cudaEventRecord(start);
        int stride = 1, in = 0, out = 1;
        while (stride < N) {
            if (USE_SHUFFLE)
                pcr_step_kernel_shuffle<<<blocks, threads>>>(d_a[in], d_b[in], d_c[in], d_d[in], d_a[out], d_b[out], d_c[out], d_d[out], N, stride);
            else
                pcr_step_kernel<<<blocks, threads>>>(d_a[in], d_b[in], d_c[in], d_d[in], d_a[out], d_b[out], d_c[out], d_d[out], N, stride);
            
            in = 1 - in; out = 1 - out;
            stride <<= 1;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_solver_time += ms;

        finalize_kernel<<<blocks, threads>>>(d_u, d_b[in], d_d[in], N);
    }

    cudaMemcpy(h_u.data(), d_u, N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Avg Solver Time per Step: " << total_solver_time/steps << " ms" << std::endl;

    // 保存 CSV
    FILE *fp = fopen("result.csv", "w");
    fprintf(fp, "x,u\n");
    for (int i = 0; i < N; i++) fprintf(fp, "%f,%f\n", i * dx, h_u[i]);
    fclose(fp);

    // 清理资源
    for(int j=0; j<2; j++) { cudaFree(d_a[j]); cudaFree(d_b[j]); cudaFree(d_c[j]); cudaFree(d_d[j]); }
    cudaFree(d_u); cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() { run_simulation(); return 0; }
