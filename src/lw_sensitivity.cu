#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>


__constant__ float c_cfl;
__constant__ int c_N;

#define CUDA_CHECK(err) if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(-1); }


__global__ void lw_naive(float* u_new, const float* u_old) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_N) return;
    int L = (i == 0) ? c_N - 1 : i - 1;
    int R = (i == c_N - 1) ? 0 : i + 1;
    float cfl2 = c_cfl * c_cfl;
    u_new[i] = u_old[i] - 0.5f * c_cfl * (u_old[R] - u_old[L]) + 0.5f * cfl2 * (u_old[R] - 2.0f * u_old[i] + u_old[L]);
}


__global__ void lw_shared(float* u_new, const float* u_old) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_N) return;

    s[tid + 1] = u_old[i];
    if (tid == 0) s[0] = u_old[(i == 0) ? c_N - 1 : i - 1];
    if (tid == blockDim.x - 1 || i == c_N - 1) s[tid + 2] = u_old[(i == c_N - 1) ? 0 : i + 1];

    __syncthreads();
    float cfl2 = c_cfl * c_cfl;
    u_new[i] = s[tid+1] - 0.5f * c_cfl * (s[tid+2] - s[tid]) + 0.5f * cfl2 * (s[tid+2] - 2.0f * s[tid+1] + s[tid]);
}


__global__ void lw_shuffle(float* u_new, const float* u_old) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_N) return;
    int lane_id = threadIdx.x & 31;
    float curr = u_old[i];

    float L = __shfl_up_sync(0xffffffff, curr, 1);
    if (lane_id == 0) L = u_old[(i == 0) ? c_N - 1 : i - 1];

    float R = __shfl_down_sync(0xffffffff, curr, 1);
    if (lane_id == 31 || i == c_N - 1) R = u_old[(i == c_N - 1) ? 0 : i + 1];

    float cfl2 = c_cfl * c_cfl;
    u_new[i] = curr - 0.5f * c_cfl * (R - L) + 0.5f * cfl2 * (R - 2.0f * curr + L);
}


float benchmark(int type, int N, int steps) {
    float h_cfl = 0.5f;
    CUDA_CHECK(cudaMemcpyToSymbol(c_cfl, &h_cfl, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_N, &N, sizeof(int)));

    float *d_u1, *d_u2;
    CUDA_CHECK(cudaMalloc(&d_u1, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u2, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_u1, 0, N * sizeof(float)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEventRecord(start);
    for (int s = 0; s < steps; ++s) {
        if (type == 0) lw_naive<<<blocks, threads>>>(d_u2, d_u1);
        else if (type == 1) lw_shared<<<blocks, threads, (threads + 2) * sizeof(float)>>>(d_u2, d_u1);
        else lw_shuffle<<<blocks, threads>>>(d_u2, d_u1);
        std::swap(d_u1, d_u2);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaFree(d_u1); cudaFree(d_u2);
    // GUpdates/s = (N * steps) / (time_ns)
    return (float)N * steps / (ms * 1e6f);
}

int main() {
    std::cout << std::setw(12) << "N" << std::setw(15) << "Naive(GU/s)" 
              << std::setw(15) << "Shared(GU/s)" << std::setw(15) << "Shuffle(GU/s)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int exp = 10; exp <= 24; exp += 2) {
        int N = 1 << exp;
        float g0 = benchmark(0, N, 100);
        float g1 = benchmark(1, N, 100);
        float g2 = benchmark(2, N, 100);
        std::cout << std::setw(12) << N << std::setw(15) << g0 << std::setw(15) << g1 << std::setw(15) << g2 << std::endl;
    }
    return 0;
}
