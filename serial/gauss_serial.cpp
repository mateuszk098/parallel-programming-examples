// Compilation: g++ -pedantic -pipe -O3 -march=native gauss_serial.cpp -o gauss_serial
// Run: ./gauss_serial.exe [1], where [1] is the matrix size power of 2.
// E.g.: ./gauss_serial.exe 15 launches calculations on 2^15 x 2^15 matrix

#include <stdlib.h>
#include <stdio.h>
#include <chrono>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// Gaussian elimination algorithm
void gaussElimination(double *matrix, int N)
{
    double ratio = 0;

    for (int i = 0; i < N; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            ratio = matrix[j * N + i] / matrix[i * N + i];

            for (int k = 0; k < N; k++)
                matrix[j * N + k] = matrix[j * N + k] - ratio * matrix[i * N + k];
        }

        ratio = matrix[i * N + i];

        for (int j = i; j < N; j++)
            matrix[i * N + j] = matrix[i * N + j] / ratio;
    }
}

// Initialize matrix
void initMatrix(double *matrix, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i * N + j] = rand() % 10;
}

// Print matrix state
void printMatrix(double *matrix, int N)
{
    printf("\n");

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%.2f    ", matrix[i * N + j]);
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    // Matrix size of 1024 x 1024
    int N = 1 << atoi(argv[1]); // (2^10)

    double *matrix = (double *)malloc(N * N * sizeof(double));

    // Initialize matrices
    initMatrix(matrix, N);

    // Print initial state
    // printMatrix(matrix, N);

    // Execution time - start
    auto start = high_resolution_clock::now();

    // Launch Gauss Elimination on CPU
    gaussElimination(matrix, N);

    // Execution time - stop
    auto stop = high_resolution_clock::now();

    // Print final state
    // printMatrix(matrix, N);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = stop - start;

    printf("\nCompleted successfully!\n");
    printf("gaussElimination() execution time on the CPU: %f ms\n", ms_double.count());

    free(matrix);

    return 0;
}