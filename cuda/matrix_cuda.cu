// N x N matrix multiplication using GPU.
// Compilation: nvcc -arch=sm_75 matrix_cuda.cu -o matrix_cuda
// GTX 1660 Ti has a compute capability of 7.5
// Run: ./matrix_cuda.exe [1], where [1] is the matrix size
// power of 2. E.g.: ./matrix_cuda.exe 15 launches calculations on 2^15 x 2^15 matrix

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>

// CUDA kernel for matrices multiplicate
__global__ void matrixMult(int *a, int *b, int *c, int N)
{
    // Compute each thread's row
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Compute each thread's column
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;

    // Boundary protection
    if ((row < N) && (col < N))
    {
        // Iterate over row, and down column
        for (int k = 0; k < N; k++)
            temp_sum += a[row * N + k] * b[k * N + col]; // Accumulate result for a single element

        // Assign result
        c[row * N + col] = temp_sum;
    }
}

// Initialize matrix
void initMatrix(int *a, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            a[i * N + j] = rand() % 100;
}

// Print matrix state
void printMatrix(int *m, int N)
{
    printf("\n");

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%d ", m[i * N + j]);
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    // Matrix size of 1024 x 1024
    int N = 1 << atoi(argv[1]); // (2^10)

    // Size in bytes of matrix
    size_t bytes = N * N * sizeof(int);

    // Host pointers
    int *h_a, *h_b, *h_c;

    // Allocate host memory
    h_a = (int *)malloc(bytes);
    h_b = (int *)malloc(bytes);
    h_c = (int *)malloc(bytes);

    // Device pointers
    int *d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize matrices
    initMatrix(h_a, N);
    initMatrix(h_b, N);

    // Print initial state
    // printMatrix(h_a, N);
    // printMatrix(h_b, N);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Threads per block
    const int BLOCK_SIZE = 16;

    // Block in each dimension
    const int GRID_SIZE = (int)ceil(N / BLOCK_SIZE);

    // Use dim3 object
    dim3 GRID(GRID_SIZE, GRID_SIZE);
    dim3 THREADS(BLOCK_SIZE, BLOCK_SIZE);

    // Execution time
    float elapsed = 0;
    cudaEvent_t start, stop;

    // Create start and stop events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);

    // Launch matrix multiplication
    matrixMult<<<GRID, THREADS>>>(d_a, d_b, d_c, N);

    // Record stop event
    cudaEventRecord(stop, 0);

    // Synchronize stop event
    cudaEventSynchronize(stop);

    // Calculate elapsed time in GPU
    cudaEventElapsedTime(&elapsed, start, stop);

    // Destroy start and stop events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy final matrix from device to Host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print final state
    // printMatrix(h_c, N);

    printf("\nCompleted successfully!\n");
    printf("matrixMult() execution time on the GPU: %f ms\n", elapsed);

    // Free Device Memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free Host Memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}