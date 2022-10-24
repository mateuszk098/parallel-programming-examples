#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <mpi.h>
#include <chrono>
#include <math.h>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

int main(int argc, char **argv)
{
    // Choice the random seed
    srand(time(NULL));

    // Total points inside circle
    int sum = 0;
    int pi_sum;

    // Iterations number
    int N = 256 * 256 * atoi(argv[1]);

    // Random coordinates from the range (0,1)
    double x, y;

    // Execution time - start
    auto start = high_resolution_clock::now();

    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    for (int i = my_rank + 1; i <= N; i += size)
    {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;

        if (x * x + y * y <= 1.0)
            sum++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&sum, &pi_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        auto stop = high_resolution_clock::now();
        double pi = pi_sum * 4.0 / N;
        // Evaluate relative error
        double err = abs(pi - acos(-1)) / pi * 100;
        /* Getting number of milliseconds as a double. */
        duration<double, std::milli> ms_double = stop - start;

        printf("\nCompleted successfully!\n");
        printf("MPI PI = %f\n", pi);
        printf("MPI relative error = %f pct\n", err);
        printf("calculatePi() execution time on the MPI: %f ms\n", ms_double.count());
    }

    MPI_Finalize();

    return 0;
}