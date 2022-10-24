// This program multiplicate N x N element matrices uses CPU
// g++ -pedantic -pipe -O3 -march=native matrix_serial.cpp -o matrix_serial
// Run: ./matrix_serial.exe [1], where [1] is the matrix size
// power of 2. E.g.: ./matrix_serial.exe 15 launches calculations on 2^15 x 2^15 matrix

#include <stdlib.h>
#include <stdio.h>
#include <chrono>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// Matrices multiplication
void matrixMult(int *a, int *b, int *c, int N)
{
    int temp_sum;

    for (int i = 0; i < N; i++)
    {
        // For every column...
        for (int j = 0; j < N; j++)
        {
            temp_sum = 0;
            // For every element in the row-column pair
            for (int k = 0; k < N; k++)
                temp_sum += a[i * N + k] * b[k * N + j]; // Accumulate the partial results
            c[i * N + j] = temp_sum;
        }
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

    // Initialize matrices
    initMatrix(h_a, N);
    initMatrix(h_b, N);

    // Print initial state
    // printMatrix(h_a, N);
    // printMatrix(h_b, N);

    // Execution time - start
    auto start = high_resolution_clock::now();

    // Launch matrix multiplication (CPU)
    matrixMult(h_a, h_b, h_c, N);

    // Execution time - stop
    auto stop = high_resolution_clock::now();

    // Print final state
    // printMatrix(h_c, N);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = stop - start;

    printf("\nCompleted successfully!\n");
    printf("matrixMult() execution time on the CPU: %f ms\n", ms_double.count());

    free(h_a);
    free(h_b);

    return 0;
}