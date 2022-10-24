#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <chrono>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// Initialize matrix
void initMatrix(int *a, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            a[i * N + j] = rand() % 100;
}

int main(int argc, char **argv)
{
    int N = 1 << atoi(argv[1]);

    int my_rank, size;
    int num_of_slaves;
    int rows, averow, extra, offset;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Status status;

    num_of_slaves = size - 1;

    int h_a[N * N], h_b[N * N], h_c[N * N];

    if (my_rank == 0)
    {

        initMatrix(h_a, N);
        initMatrix(h_b, N);

        // Execution time - start
        auto start = high_resolution_clock::now();

        averow = N / num_of_slaves;
        extra = N % num_of_slaves;
        offset = 0;

        // sending
        for (int destination = 1; destination <= num_of_slaves; destination++)
        {
            rows = (destination <= extra) ? averow + 1 : averow;
            MPI_Send(&offset, 1, MPI_INT, destination, 999, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, destination, 999, MPI_COMM_WORLD);
            MPI_Send(&h_a[offset], rows * N, MPI_INT, destination, 999, MPI_COMM_WORLD);
            MPI_Send(&h_b, N * N, MPI_INT, destination, 999, MPI_COMM_WORLD);
            offset = offset + rows;
        }

        // receiving
        for (int i = 1; i <= num_of_slaves; i++)
        {
            MPI_Recv(&offset, 1, MPI_INT, i, 555, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, i, 555, MPI_COMM_WORLD, &status);
            MPI_Recv(&h_c[offset], rows * N, MPI_INT, i, 555, MPI_COMM_WORLD, &status);
        }

        // Execution time - stop
        auto stop = high_resolution_clock::now();
        /* Getting number of milliseconds as a double. */
        duration<double, std::milli> ms_double = stop - start;

        printf("\nCompleted successfully!\n");
        printf("matrixMult() execution time on the MPI: %f ms\n", ms_double.count());
    }

    else
    {
        MPI_Recv(&offset, 1, MPI_INT, 0, 999, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, 0, 999, MPI_COMM_WORLD, &status);
        MPI_Recv(&h_a, rows * N, MPI_INT, 0, 999, MPI_COMM_WORLD, &status);
        MPI_Recv(&h_b, N * N, MPI_INT, 0, 999, MPI_COMM_WORLD, &status);

        for (int k = 0; k < N; k++)
        {
            for (int i = 0; i < rows; i++)
            {
                h_c[i * rows + k] = 0.0;
                for (int j = 0; j < N; j++)
                    h_c[i * rows + k] = h_c[i * rows + k] + h_a[i * rows + j] * h_b[j * N + k];
            }
        }

        MPI_Send(&offset, 1, MPI_INT, 0, 555, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 555, MPI_COMM_WORLD);
        MPI_Send(&h_c, rows * N, MPI_INT, 0, 555, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}