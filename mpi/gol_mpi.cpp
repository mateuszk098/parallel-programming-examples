#include <iostream>
#include <cstdlib>
#include <fstream>
#include <mpi.h>
#include <chrono>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

void initMatrix(char *m, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            m[i * N + j] = rand() % 2 > 0 ? '.' : 'X';
}

char analyzeCell(char *c_m, int N, int i, int j)
{
    int alive_neighbours = 0, dead_neighbours = 0;

    c_m[(i - 1) * N + (j - 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
    c_m[(i - 1) * N + j] == '.' ? dead_neighbours++ : alive_neighbours++;
    c_m[(i - 1) * N + (j + 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
    c_m[i * N + (j + 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
    c_m[(i + 1) * N + (j + 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
    c_m[(i + 1) * N + j] == '.' ? dead_neighbours++ : alive_neighbours++;
    c_m[(i + 1) * N + (j - 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
    c_m[i * N + (j - 1)] == '.' ? dead_neighbours++ : alive_neighbours++;

    if (alive_neighbours < 2)
        return '.';

    else if (alive_neighbours > 3)
        return '.';

    else if (c_m[i * N + j] == 'X' && (alive_neighbours == 2 || alive_neighbours == 3))
        return 'X';

    else if (c_m[i * N + j] == '.' && alive_neighbours == 3)
        return 'X';

    else
        return 'X';
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    double start, stop;

    // Matrix size
    int N = 1 << atoi(argv[1]);

    if (N % size != 0)
    {
        printf("\nThis application must be run with size that the rest of the division N/size is equal to 0!\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    N = N + 2;

    // Matrix size in bytes
    size_t bytes = N * N * (sizeof(char));

    // Host pointers to current matrix and new matrix
    char *c_m, *n_m;

    // Allocatee host memory
    c_m = (char *)malloc(bytes);
    n_m = (char *)malloc(bytes);

    initMatrix(c_m, N);

    if (my_rank == 0)
    {
        // Execution time - start
        start = MPI_Wtime();
    }

    int my_size = (N - 2) / size;
    int size_of_slice = my_size + 2;

    char *my_slice = new char[size_of_slice * N];

    for (int i = 0; i < size_of_slice; i++)
    {
        for (int j = 0; j < N; j++)
        {
            my_slice[i * N + j] = c_m[i * N + j + my_size * my_rank];
        }
    }

    char *my_new_slice = new char[my_size * N];

    for (int i = 1; i < size_of_slice - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
            my_new_slice[i * N + j] = analyzeCell(my_slice, N, i, j);
    }

    // BARRIER TO TRACK WHEN CALC ARE DONE
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(&my_new_slice, (N - 2) * size_of_slice, MPI_CHAR, n_m, (N - 2) * size_of_slice, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        // Execution time - stop
        stop = MPI_Wtime();
        /* Getting number of milliseconds as a double. */
        double ms_double = stop - start;
        printf("\nCompleted successfully!\n");
        printf("Game of life execution time with MPI: %f ms\n", ms_double);
    }

    return 0;
}