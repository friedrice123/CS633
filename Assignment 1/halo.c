#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 6) {
        if (rank == 0) {
            printf("Usage: %s Px N_side num_time_steps seed stencil\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Parsing arguments for the side length of the data points square
    int N_side = atoi(argv[2]); // Side length of the data points square
    int num_time_steps = atoi(argv[3]);
    int seed = atoi(argv[4]);
    int stencil = atoi(argv[5]); // Stencil type: 5-point or 9-point

    // Total number of data points per process
    int total_data_points = N_side * N_side;

    // Initialize data points with seed
    srand(seed * (rank + 10));
    double* data = (double*)malloc(total_data_points * sizeof(double));
    for (int i = 0; i < N_side; ++i) {
        for (int j = 0; j < N_side; ++j) {
            data[i * N_side + j] = fabs((rand() + (i * rand()) + (j * rank)) / 100.0);
        }
    }

    // Check for correct process grid size
    if (PX * PY != size) {
        if (rank == 0) printf("Error: Px * Py does not match the number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    // Parsing arguments
    int N = atoi(argv[1]); // Square root of total data points per process
    int num_time_steps = atoi(argv[2]);
    int mode = atoi(argv[3]); // Mode for halo exchange

    // Initialization of data points
    double* data = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            data[i * N + j] = fabs(rand() / (double)RAND_MAX);
        }
    }

    // Buffer for MPI_Pack and MPI_Unpack
    int buffer_size = N * sizeof(double) + MPI_BSEND_OVERHEAD;
    char* buffer_send = (char*)malloc(buffer_size);
    char* buffer_recv = (char*)malloc(buffer_size);

    // Assuming a simple 2D grid of processes (Px x Py)
    int Px = sqrt(size); // Example: process grid dimensions
    int Py = size / Px;
    if (Px * Py != size) {
        if (rank == 0) printf("Error: Px * Py does not match the number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    // Calculate position in the grid
    int x = rank % Px;
    int y = rank / Px;

    // Flags to check for neighbors
    int has_top = y > 0;
    int has_bottom = y < (Py - 1);
    // Left and right neighbors can be added similarly

    MPI_Request requests[8]; // For non-blocking communication
    int request_count = 0;

    for (int step = 0; step < num_time_steps; ++step) {
        if (mode == 2) { // Only focusing on mode 2
            int position;

            // Pack and send to top neighbor
            if (has_top) {
                position = 0;
                MPI_Pack(data, N, MPI_DOUBLE, buffer_send, buffer_size, &position, MPI_COMM_WORLD);
                MPI_Isend(buffer_send, position, MPI_PACKED, rank - Px, 0, MPI_COMM_WORLD, &requests[request_count++]);
            }

            // Receive from top neighbor
            if (has_top) {
                MPI_Irecv(buffer_recv, buffer_size, MPI_PACKED, rank - Px, 0, MPI_COMM_WORLD, &requests[request_count++]);
            }

            // Wait for all communications to complete
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

            // Unpack received data from top neighbor
            if (has_top) {
                position = 0;
                MPI_Unpack(buffer_recv, buffer_size, &position, /* Appropriate target buffer */, N, MPI_DOUBLE, MPI_COMM_WORLD);
                // Update data based on received data
            }

            // Reset request count for next iteration
            request_count = 0;
        }
        // Additional logic for stencil computation and other modes can be added here
    }

    free(data);
    free(buffer_send);
    free(buffer_recv);
    MPI_Finalize();
    return 0;
}