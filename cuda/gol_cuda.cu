// Simulate game of life using GPU.
// Compilation: nvcc -arch=sm_75 gol_cuda.cu -o gol_cuda
// GTX 1660 Ti has a compute capability of 7.5
// Run: ./gol_cuda.exe [1], where [1] is the matrix size
// power of 2. E.g.: ./gol_cuda.exe 15 launches calculations on 2^15 x 2^15 matrix

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void analyzeCell(char *c_m, char *n_m, int N)
{
    int alive_neighbours = 0, dead_neighbours = 0;

    // Compute each thread's row
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    // Compute each thread's column
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if ((i < N - 1) && (j < N - 1))
    {
        c_m[(i - 1) * N + (j - 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
        c_m[(i - 1) * N + j] == '.' ? dead_neighbours++ : alive_neighbours++;
        c_m[(i - 1) * N + (j + 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
        c_m[i * N + (j + 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
        c_m[(i + 1) * N + (j + 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
        c_m[(i + 1) * N + j] == '.' ? dead_neighbours++ : alive_neighbours++;
        c_m[(i + 1) * N + (j - 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
        c_m[i * N + (j - 1)] == '.' ? dead_neighbours++ : alive_neighbours++;

        if (alive_neighbours < 2)
            n_m[i * N + j] = '.';

        else if (alive_neighbours > 3)
            n_m[i * N + j] = '.';

        else if (c_m[i * N + j] == 'X' && (alive_neighbours == 2 || alive_neighbours == 3))
            n_m[i * N + j] = 'X';

        else if (c_m[i * N + j] == '.' && alive_neighbours == 3)
            n_m[i * N + j] = 'X';

        else
            n_m[i * N + j] = 'X';
    }
}

// Initialize matrix
void initMatrix(char *m, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            m[i * N + j] = (rand() % 2 > 0 ? '.' : 'X');
}

// Print matrix state
void printMatrix(char *m, int N)
{
    printf("\n");

    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
            printf("%c ", m[i * N + j]);
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    // Matrix size
    int N = 1 << atoi(argv[1]);

    // Matrix size in bytes
    size_t bytes = N * N * sizeof(char);

    // Host pointers to current matrix and new matrix
    char *h_c_m, *h_n_m;

    // Allocatee host memory
    h_c_m = (char *)malloc(bytes);
    h_n_m = (char *)malloc(bytes);

    // Device pointers
    char *d_c_m, *d_n_m;

    // Allocate device memory
    cudaMalloc(&d_c_m, bytes);
    cudaMalloc(&d_n_m, bytes);

    // Initialize matrix
    initMatrix(h_c_m, N);

    // Print initial state
    // printMatrix(h_c_m, N);

    // Copy current matrix from host to device
    cudaMemcpy(d_c_m, h_c_m, bytes, cudaMemcpyHostToDevice);

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

    // Launch Game of Life (GPU)
    analyzeCell<<<GRID, THREADS>>>(d_c_m, d_n_m, N);

    // Record stop event
    cudaEventRecord(stop, 0);

    // Synchronize stop event
    cudaEventSynchronize(stop);

    // Calculate elapsed time in GPU
    cudaEventElapsedTime(&elapsed, start, stop);

    // Destroy start and stop events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy new matrix from device to host
    cudaMemcpy(h_n_m, d_n_m, bytes, cudaMemcpyDeviceToHost);

    printf("\nCompleted successfully!\n");
    printf("analyzeCell() execution time on the GPU: %f ms\n", elapsed);

    // Print final state
    // printMatrix(h_n_m, N);

    // Free Device Memory
    cudaFree(d_c_m);
    cudaFree(d_n_m);

    // Free Host Memory
    free(h_c_m);
    free(h_n_m);

    return 0;
}
