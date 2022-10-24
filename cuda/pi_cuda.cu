// Calculate PI number using GPU and Monte Carlo method.
// Compilation: nvcc -arch=sm_75 pi_cuda.cu -o pi_cuda
// GTX 1660 Ti has a compute capability of 7.5
// Run: ./pi_cuda.exe [1], where [1] is the special iteration number
// but the whole number of iterations is 256 * 256 * [1], so be careful.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <stdio.h>

#define SHMEM 256

// CUDA kernel to evaluate the PI number
__global__ void calculatePi(curandState *state, int *count, int M)
{
    // Calculate global thread id
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

    // Initialize kernel random seed
    curand_init(123456789, index, 0, &state[index]);

    // Allocate shared memory
    __shared__ int cache[SHMEM];
    cache[threadIdx.x] = 0;
    __syncthreads();

    // Random numbers to evaluate PI
    double x, y;

    // Circle condition loop
    for (int i = 0; i < M; i++)
    {
        x = curand_uniform(&state[index]);
        y = curand_uniform(&state[index]);

        if (x * x + y * y <= 1)
            cache[threadIdx.x]++;
    }

    // Reduction
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
            cache[threadIdx.x] += cache[threadIdx.x + i];

        i /= 2;
        __syncthreads();
    }

    // Update to our global variable count
    if (threadIdx.x == 0)
        atomicAdd(count, cache[0]);
}

int main(int argc, char **argv)
{
    unsigned int N = SHMEM * SHMEM;
    unsigned int M = atoi(argv[1]);
    size_t bytes = N * sizeof(int);

    // Host pointer count
    int *h_count;

    // Device pointer count
    int *d_count;

    // Device curand pointer
    curandState *d_state;

    // Allocate host memory
    h_count = (int *)malloc(N * bytes);

    // Allocate device memory
    cudaMalloc(&d_count, bytes);
    cudaMalloc(&d_state, N * sizeof(curandState));

    // Threads per block
    const int BLOCK_SIZE = SHMEM;

    // Block in each dimension
    const int GRID_SIZE = (int)ceil(N / BLOCK_SIZE);

    // Execution time
    float elapsed = 0;
    cudaEvent_t start, stop;

    // Create start and stop events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);

    // Launch CUDA kernel
    calculatePi<<<GRID_SIZE, BLOCK_SIZE>>>(d_state, d_count, M);

    // Record stop event
    cudaEventRecord(stop, 0);

    // Synchronize stop event
    cudaEventSynchronize(stop);

    // Calculate elapsed time in GPU
    cudaEventElapsedTime(&elapsed, start, stop);

    // Destroy start and stop events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy results from device to Host
    cudaMemcpy(h_count, d_count, bytes, cudaMemcpyDeviceToHost);

    // Calculate PI
    double pi = *h_count * 4.0 / (N * M);

    // Evaluate relative error
    double err = abs(pi - acos(-1)) / pi * 100;

    printf("\nCompleted successfully!\n");
    printf("GPU PI = %f\n", pi);
    printf("GPU relative error = %f pct\n", err);
    printf("calculatePi() execution time on the GPU: %f ms\n", elapsed);

    // Free device memory
    cudaFree(d_count);
    cudaFree(d_state);

    // Free host memory
    free(h_count);

    return 0;
}