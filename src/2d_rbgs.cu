#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ----------------------------------------------------------------
// 1. Naive 版本: 简单的 if-else 判定 (严重 Warp Divergence)
// ----------------------------------------------------------------
__global__ void rbgs_naive(float* u, int W, int H, int color) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;

    if (i > 0 && i < H - 1 && j > 0 && j < W - 1) {
        if ((i + j) % 2 == color) { 
            int idx = i * W + j;
            u[idx] = 0.25f * (u[idx - W] + u[idx + W] + u[idx - 1] + u[idx + 1]);
        }
    }
}
/*
__global__ void rbgs_naive(float* u, int W, int H, int color){
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;

    if (i > 0 && i < H - 1 && j > 0 && j < W - 1){
        if ((i + j) % 2 == color){ //warp divergence
            int idx = i * W + j;
            u[idx] = 0.25f * (u[idx - W] + u[idx + W] + u[idx - 1] + u[idx + 1]);
        }
    }
}
*/
/*compressed (if versioin)
__global__ void rbgs_compressed(float* u, int W, int H, int color){
    int j_half = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;

    int half_W = W / 2;

    if (i > 0 && i < H - 1 && j_half > 0 && j_half < half_W){
        if (color == 0){
            int pad_row = (i % 2 == 0) ? 2*j_half : 2*j_half + 1;
            int idx = i * W + pad_row;
            u[idx] = 0.25f * (u[idx - W] + u[idx + W] + u[idx - 1] + u[idx + 1]);
        }
        if (color == 1){
            int pad_row = (i % 2 == 0) ? 2*j_half + 1 : 2*j_half;
            int idx = i * W + pad_row;
            u[idx] = 0.25f * (u[idx - W] + u[idx + W] + u[idx - 1] + u[idx + 1]);

        }
    }
}
*/
// ----------------------------------------------------------------
// 2. Compressed 版本: 解决 Warp Divergence (线程利用率 100%)
// ----------------------------------------------------------------
__global__ void rbgs_compressed(float* u, int W, int H, int color) {
    int j_half = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;

    int half_W = W / 2;
    if (i > 0 && i < H - 1 && j_half < half_W) {
        int j = 2 * j_half + ((i + color) & 1); //Bitwise Optimization  奇行的红色或者偶行的黑色，需要偏移一个位置
        
        if (j > 0 && j < W - 1) {
            int idx = i * W + j;
            u[idx] = 0.25f * (u[idx - W] + u[idx + W] + u[idx - 1] + u[idx + 1]);
        }
    }
}

// ----------------------------------------------------------------
// 3. Split Layout 版本: 解决 Memory Coalescing (极致访存效率)
// ----------------------------------------------------------------
__global__ void rbgs_split(float* this_color, const float* other_color, int half_W, int H, int color_step) {
    int j_half = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;

    if (i > 0 && i < H - 1 && j_half > 0 && j_half < half_W - 1) {
        int idx = i * half_W + j_half;
        int row_parity = i & 1;
        
        // 核心偏移逻辑：根据行奇偶性寻找在另一个数组里的邻居
        int left = idx - row_parity + color_step;
        //int left = idx - 1 + ((row_parity + color_setp) & 1)   这个更容易理解，idx-1是基准，然后是否偏移取决于偶行的黑元素(0+1),或奇行的红元素(1+0)
        int right = left + 1;
        
        this_color[idx] = 0.25f * (other_color[idx - half_W] + other_color[idx + half_W] + 
                                   other_color[left] + other_color[right]);
    }
}

// 计时辅助函数
float test_performance(int version, int W, int H, int iterations) {
    float *d_u, *d_red, *d_black;
    size_t size = W * H * sizeof(float);
    size_t half_size = (W / 2) * H * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (version == 3) {
        cudaMalloc(&d_red, half_size);
        cudaMalloc(&d_black, half_size);
        cudaMemset(d_red, 0, half_size);
        cudaMemset(d_black, 0, half_size);
    } else {
        cudaMalloc(&d_u, size);
        cudaMemset(d_u, 0, size);
    }

    dim3 block(32, 16);
    
    cudaEventRecord(start);
    for (int it = 0; it < iterations; it++) {
        if (version == 1) {
            dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
            rbgs_naive<<<grid, block>>>(d_u, W, H, 0);
            rbgs_naive<<<grid, block>>>(d_u, W, H, 1);
        } else if (version == 2) {
            dim3 grid((W / 2 + block.x - 1) / block.x, (H + block.y - 1) / block.y);
            rbgs_compressed<<<grid, block>>>(d_u, W, H, 0);
            rbgs_compressed<<<grid, block>>>(d_u, W, H, 1);
        } else if (version == 3) {
            dim3 grid((W / 2 + block.x - 1) / block.x, (H + block.y - 1) / block.y);
            rbgs_split<<<grid, block>>>(d_red, d_black, W / 2, H, 0);
            rbgs_split<<<grid, block>>>(d_black, d_red, W / 2, H, 1);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (version == 3) { cudaFree(d_red); cudaFree(d_black); }
    else { cudaFree(d_u); }
    
    return milliseconds;
}

int main() {
    std::vector<int> sizes = {512, 1024, 2048, 4096, 8192};
    int iterations = 100;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Grid Size | Naive (ms) | Compressed (ms) | Split (ms) | Speedup (Split vs Naive)\n";
    std::cout << "---------------------------------------------------------------------------\n";

    for (int N : sizes) {
        float t1 = test_performance(1, N, N, iterations);
        float t2 = test_performance(2, N, N, iterations);
        float t3 = test_performance(3, N, N, iterations);
        
        std::cout << N << "x" << N << "   | " 
                  << t1 << "    | " << t2 << "        | " << t3 << "    | " 
                  << t1 / t3 << "x\n";
    }

    return 0;
}
