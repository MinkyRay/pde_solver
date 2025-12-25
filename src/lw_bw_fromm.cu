#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <string>

#define CUDA_CHECK(err) if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(-1); }


__global__ void advection_lax_wendroff_kernel(float* u_new, const float* u_old, int n, float cfl) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int L1 = (i == 0) ? n - 1 : i - 1;
        int R1 = (i == n - 1) ? 0 : i + 1;
        float cfl2 = cfl * cfl;
        u_new[i] = u_old[i] - 0.5f * cfl * (u_old[R1] - u_old[L1]) + 0.5f * cfl2 * (u_old[R1] - 2.0f * u_old[i] + u_old[L1]);
    }
}
/*
__global__ void advection_lax_wendroff_kernel_warp_shuffle(float* u_new, const float* u_old, int n, float cfl) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val_left, val_right;
        int lane_id = tid % 32;
        float val_curr = u_old[i];

        //left shuffle
        if (lane_id == 0){
            int left_idx = (i == 0) ? n - 1 : i - 1;
            val_left = u_old[left_idx];
        }else{
            val_left = __shfl_up_sync(0xffffffff, val_curr, 1);
        }
        //right shuffle
        if (lane_id == 31 || i == n - 1){
            int right_idx = (i == n - 1) ? 0 : i + 1;
            val_right = u_old[right_idx];
        }else{
            val_right = __shfl_down_sync(0xffffffff, val_curr, 1);
        }
        float cfl2 = cfl * cfl;
        u_new[i] = val_curr - 0.5f * cfl * (val_right - val_left) + 0.5f * cfl2 * (val_right - 2.0f * val_curr + val_left);
    }
}
*/


__global__ void advection_beam_warming_kernel(float* u_new, const float* u_old, int n, float cfl) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int L1 = (i == 0) ? n - 1 : i - 1;
        int L2 = (i == 0) ? n - 2 : (i == 1 ? n - 1 : i - 2);
        float cfl2 = cfl * cfl;
        u_new[i] = u_old[i] - 0.5f * cfl * (3.0f * u_old[i] - 4.0f * u_old[L1] + u_old[L2]) + 0.5f * cfl2 * (u_old[i] - 2.0f * u_old[L1] + u_old[L2]);
    }
}




__global__ void advection_fromm_kernel(float* u_new, const float* u_old, int n, float cfl) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int L1 = (i == 0) ? n - 1 : i - 1;
        int R1 = (i == n - 1) ? 0 : i + 1;
        int L2 = (i == 0) ? n - 2 : (i == 1 ? n - 1 : i - 2);
        
        float cfl2 = cfl * cfl;

        float lw = u_old[i] - 0.5f * cfl * (u_old[R1] - u_old[L1]) + 0.5f * cfl2 * (u_old[R1] - 2.0f * u_old[i] + u_old[L1]);
        float bw = u_old[i] - 0.5f * cfl * (3.0f * u_old[i] - 4.0f * u_old[L1] + u_old[L2]) + 0.5f * cfl2 * (u_old[i] - 2.0f * u_old[L1] + u_old[L2]);
        
        u_new[i] = 0.5f * (lw + bw);
    }
}

void save_to_file(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename);
    for (float val : data) file << val << "\n";
    file.close();
}

void run_simulation(int type, int N, int STEPS, float cfl, const std::vector<float>& init, const std::string& name) {
    size_t size = N * sizeof(float);
    float *d_u1, *d_u2;
    cudaMalloc(&d_u1, size); cudaMalloc(&d_u2, size);
    cudaMemcpy(d_u1, init.data(), size, cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float *d_old = d_u1, *d_new = d_u2;

    for (int s = 0; s < STEPS; ++s) {
        if (type == 0) advection_lax_wendroff_kernel<<<blocks, threads>>>(d_new, d_old, N, cfl);
        else if (type == 1) advection_beam_warming_kernel<<<blocks, threads>>>(d_new, d_old, N, cfl);
        else advection_fromm_kernel<<<blocks, threads>>>(d_new, d_old, N, cfl);
        std::swap(d_old, d_new);
    }
    
    std::vector<float> res(N);
    cudaMemcpy(res.data(), d_old, size, cudaMemcpyDeviceToHost);
    save_to_file(name + ".txt", res);
    cudaFree(d_u1); cudaFree(d_u2);
}

int main() {
    const int N = 1024;
    const int STEPS = 600;
    const float cfl = 0.5f;
    std::vector<float> init(N);
    for (int i = 0; i < N; ++i) init[i] = ((i*1.0f/N) > 0.2f && (i*1.0f/N) < 0.4f) ? 1.0f : 0.0f;

    save_to_file("initial.txt", init);
    std::cout << "Simulating LW..." << std::endl; run_simulation(0, N, STEPS, cfl, init, "lw");
    std::cout << "Simulating BW..." << std::endl; run_simulation(1, N, STEPS, cfl, init, "bw");
    std::cout << "Simulating Fromm..." << std::endl; run_simulation(2, N, STEPS, cfl, init, "fromm");

    std::cout << "All done! Use Python to plot 'initial.txt', 'lw.txt', 'bw.txt', 'fromm.txt'." << std::endl;
    return 0;
}
