#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

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

    int Px = atoi(argv[1]); // Number of processes in the x-direction
    int N = atoi(argv[2]);  // Size of the data matrix

    // Newton-Raphson method to calculate the square root of N
    int sqrt1 = N / 2;  
    int temp = 0;  
 
    while (sqrt1 != temp){  
        temp = sqrt1;
        sqrt1 = ( N / temp + temp) / 2;  
    }

    int N_side = sqrt1;                      // Assuming N is a perfect square
    int num_time_steps = atoi(argv[3]);      // Number of time steps
    int seed = atoi(argv[4]);                // Seed for random number generator
    int stencil = atoi(argv[5]);             // 5-point or 9-point

    srand(seed * (rank + 10));
    // Creating the data matrix
    double** data = (double**)malloc(N_side * sizeof(double*));
    double** data_new = (double**)malloc(N_side * sizeof(double*));
    
    for (double i = 0; i < N_side; i++){
        data[(int)i] = (double*)malloc(N_side * sizeof(double));
        data_new[(int)i] = (double*)malloc(N_side * sizeof(double));
    }
    for (double i = 0; i < N_side; i++) {
        for (double j = 0; j < N_side; j++) {
            data[(int)i][(int)j] = abs((rand() + (i * rand()) + (j * rank)) / 100.0);
        }
    }

    int Py = size / Px;
    if (Px * Py != size) {
        if (rank == 0) printf("Error: Px * Py does not match the number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    // Calculating position of the process in the processes grid
    int x = rank % Px;
    int y = rank / Px;

    // Which neighbours exist
    int has_top = y > 0;
    int has_bottom = y < (Py - 1);
    int has_left = x > 0;
    int has_right = x < (Px - 1);


    MPI_Request requests[4];
    int request_count = 0;

    MPI_Status status[4];
    int status_count = 0;

    double sTime, eTime;
    sTime = MPI_Wtime();   // Start the timer

    for (int step = 0; step < num_time_steps; step++) {
        if(stencil == 5) {
            double top_buff_send[N_side], top_buff_recv[N_side];
            double bottom_buff_send[N_side], bottom_buff_recv[N_side];
            double left_buff_send[N_side], left_buff_recv[N_side]; 
            double right_buff_send[N_side], right_buff_recv[N_side];
            double top[N_side], bottom[N_side], left[N_side], right[N_side];
            int position;

            // Pack and send to top neighbor
            if (has_top) {
                position = 0;
                for(int i = 0; i < N_side; i++){
                    MPI_Pack(&data[0][i], 1, MPI_DOUBLE, top_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                }
                MPI_Isend(top_buff_send, position, MPI_PACKED, rank - Px, rank, MPI_COMM_WORLD, &requests[request_count++]);
            }

            // Receive from top neighbor
            if (has_top) {
                position = 0;
                MPI_Recv(top_buff_recv, N_side*(sizeof(double)), MPI_PACKED, rank - Px, rank - Px, MPI_COMM_WORLD, &status[status_count++]);
                
                MPI_Wait(&requests[request_count-1], &status[status_count-1]); // Waiting for the communication to complete
                for(int i = 0; i < N_side; i++){
                    MPI_Unpack(top_buff_recv, N_side*(sizeof(double)), &position, top+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                }
            }

            // Pack and send to bottom neighbor
            if (has_bottom) {
                position = 0;
                for(int i = 0; i < N_side; i++){
                    MPI_Pack(&data[N_side-1][i], 1, MPI_DOUBLE, bottom_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                }
                MPI_Isend(bottom_buff_send, position, MPI_PACKED, rank + Px, rank, MPI_COMM_WORLD, &requests[request_count++]);
            }

            // Receive from bottom neighbor
            if (has_bottom) {
                position = 0;
                MPI_Recv(bottom_buff_recv, N_side*(sizeof(double)), MPI_PACKED, rank + Px, rank + Px, MPI_COMM_WORLD, &status[status_count++]);
                
                MPI_Wait(&requests[request_count-1], &status[status_count-1]); // Waiting for the communication to complete
                for(int i = 0; i < N_side; i++){
                    MPI_Unpack(bottom_buff_recv, N_side*(sizeof(double)), &position, bottom+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                }
            }

            // Pack and send to left neighbor
            if (has_left) {
                position = 0;
                for(int i = 0; i < N_side; i++){
                    MPI_Pack(&data[i][0], 1, MPI_DOUBLE, left_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                }
                MPI_Isend(left_buff_send, position, MPI_PACKED, rank - 1, rank, MPI_COMM_WORLD, &requests[request_count++]);
            }

            // Receive from left neighbor
            if (has_left) {
                position = 0;
                MPI_Recv(left_buff_recv, N_side*(sizeof(double)), MPI_PACKED, rank - 1, rank - 1, MPI_COMM_WORLD, &status[status_count++]);

                MPI_Wait(&requests[request_count-1], &status[status_count-1]); // Waiting for the communication to complete
                for(int i = 0; i < N_side; i++){
                    MPI_Unpack(left_buff_recv, N_side*(sizeof(double)), &position, left+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                }
            }

            // Pack and send to right neighbor
            if (has_right) {
                position = 0;
                for(int i = 0; i < N_side; i++){
                    MPI_Pack(&data[i][N_side-1], 1, MPI_DOUBLE, right_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                }
                MPI_Isend(right_buff_send, position, MPI_PACKED, rank + 1, rank, MPI_COMM_WORLD, &requests[request_count++]);
            }

            // Receive from right neighbor
            if (has_right) {
                position = 0;
                MPI_Recv(right_buff_recv, N_side*(sizeof(double)), MPI_PACKED, rank + 1, rank + 1, MPI_COMM_WORLD, &status[status_count++]);
            
                MPI_Wait(&requests[request_count-1], &status[status_count-1]); // Waiting for the communication to complete
                for(int i = 0; i < N_side; i++){
                    MPI_Unpack(right_buff_recv, N_side*(sizeof(double)), &position, right+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                }
            }

            // Update the centre data points
            for (int i = 1; i < N_side - 1; i++) {
                for (int j = 1; j < N_side - 1; j++) {
                    data_new[i][j] = (data[i][j] + data[i-1][j] + data[i+1][j] + data[i][j-1] + data[i][j+1]) / 5.0;
                }
            }

            // Update the halo points
            if (has_top) {
                for (int i = 1; i < N_side - 1; i++) {
                    data_new[0][i] = (data[0][i] + top[i] + data[1][i] + data[0][i-1] + data[0][i+1]) / 5.0;
                }
            }

            if (has_bottom) {
                for (int i = 1; i < N_side - 1; i++) {
                    data_new[N_side-1][i] = (data[N_side-1][i] + bottom[i] + data[N_side-2][i] + data[N_side-1][i-1] + data[N_side-1][i+1]) / 5.0;
                }
            }

            if (has_left) {
                for (int i = 1; i < N_side - 1; i++) {
                    data_new[i][0] = (data[i][0] + left[i] + data[i+1][0] + data[i-1][0] + data[i][1]) / 5.0;
                }
            }

            if (has_right) {
                for (int i = 1; i < N_side - 1; i++) {
                    data_new[i][N_side-1] = (data[i][N_side-1] + right[i] + data[i+1][N_side-1] + data[i-1][N_side-1] + data[i][N_side-2]) / 5.0;
                }
            }

            // Update the inner corner points
            if (has_top && has_left) {
                data_new[0][0] = (data[0][0] + top[0] + data[1][0] + left[0] + data[0][1]) / 5.0;
            }

            if (has_top && has_right) {
                data_new[0][N_side-1] = (data[0][N_side-1] + top[N_side-1] + data[1][N_side-1] + data[0][N_side-2] + right[0]) / 5.0;
            }

            if (has_bottom && has_left) {
                data_new[N_side-1][0] = (data[N_side-1][0] + bottom[0] + data[N_side-2][0] + left[N_side-1] + data[N_side-1][1]) / 5.0;
            }

            if (has_bottom && has_right) {
                data_new[N_side-1][N_side-1] = (data[N_side-1][N_side-1] + bottom[N_side-1] + data[N_side-2][N_side-1] + data[N_side-1][N_side-2] + right[N_side-1]) / 5.0;
            }

            // Update the edge points
            if(!has_top){
                for(int i = 1; i < N_side - 1; i++){
                    data_new[0][i] = (data[0][i] + data[1][i] + data[0][i-1] + data[0][i+1]) / 4.0;
                }
            }

            if(!has_bottom){
                for(int i = 1; i < N_side - 1; i++){
                    data_new[N_side-1][i] = (data[N_side-1][i] + data[N_side-2][i] + data[N_side-1][i-1] + data[N_side-1][i+1]) / 4.0;
                }
            }

            if(!has_left){
                for(int i = 1; i < N_side - 1; i++){
                    data_new[i][0] = (data[i][0] + data[i+1][0] + data[i-1][0] + data[i][1]) / 4.0;
                }
            }

            if(!has_right){
                for(int i = 1; i < N_side - 1; i++){
                    data_new[i][N_side-1] = (data[i][N_side-1] + data[i+1][N_side-1] + data[i-1][N_side-1] + data[i][N_side-2]) / 4.0;
                }
            }
                
            // Update the outer corner points
            if(!has_top && !has_left){
                data_new[0][0] = (data[0][0] + data[1][0] + data[0][1]) / 3.0;
            }

            if(!has_top && !has_right){
                data_new[0][N_side-1] = (data[0][N_side-1] + data[1][N_side-1] + data[0][N_side-2]) / 3.0;
            }

            if(!has_bottom && !has_left){
                data_new[N_side-1][0] = (data[N_side-1][0] + data[N_side-2][0] + data[N_side-1][1]) / 3.0;
            }

            if(!has_bottom && !has_right){
                data_new[N_side-1][N_side-1] = (data[N_side-1][N_side-1] + data[N_side-2][N_side-1] + data[N_side-1][N_side-2]) / 3.0;
            }

            if(has_top && !has_left){
                data_new[0][0] = (data[0][0] + top[0] + data[1][0] + data[0][1]) / 4.0;
            }

            if(has_top && !has_right){
                data_new[0][N_side-1] = (data[0][N_side-1] + top[N_side-1] + data[1][N_side-1] + data[0][N_side-2]) / 4.0;
            }

            if(has_bottom && !has_left){
                data_new[N_side-1][0] = (data[N_side-1][0] + bottom[0] + data[N_side-2][0] + data[N_side-1][1]) / 4.0;
            }

            if(has_bottom && !has_right){
                data_new[N_side-1][N_side-1] = (data[N_side-1][N_side-1] + bottom[N_side-1] + data[N_side-2][N_side-1] + data[N_side-1][N_side-2]) / 4.0;
            }

            if(!has_top && has_left){
                data_new[0][0] = (data[0][0] + data[1][0] + left[0] + data[0][1]) / 4.0;
            }

            if(!has_top && has_right){
                data_new[0][N_side-1] = (data[0][N_side-1] + data[1][N_side-1] + right[0] + data[0][N_side-2]) / 4.0;
            }

            if(!has_bottom && has_left){
                data_new[N_side-1][0] = (data[N_side-1][0] + data[N_side-2][0] + left[N_side-1] + data[N_side-1][1]) / 4.0;
            }

            if(!has_bottom && has_right){
                data_new[N_side-1][N_side-1] = (data[N_side-1][N_side-1] + data[N_side-2][N_side-1] + right[N_side-1] + data[N_side-1][N_side-2]) / 4.0;
            }

            data = data_new;

            // Reset request and status count for next iteration
            request_count = 0;
            status_count = 0;
        }
        else if(stencil == 9){
            double top_buff_send[2*N_side], top_buff_recv[2*N_side];
            double bottom_buff_send[2*N_side], bottom_buff_recv[2*N_side];
            double left_buff_send[2*N_side], left_buff_recv[2*N_side];
            double right_buff_send[2*N_side], right_buff_recv[2*N_side];
            double top[2*N_side], bottom[2*N_side], left[2*N_side], right[2*N_side];
            int position;

            // Pack and send to top neighbor
            if (has_top) {
                position = 0;
                for(int i = 0; i < N_side; i++){
                    MPI_Pack(&data[0][i], 1, MPI_DOUBLE, top_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                }
                for(int i = 0; i < N_side; i++){
                    MPI_Pack(&data[1][i], 1, MPI_DOUBLE, top_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                }
                MPI_Isend(top_buff_send, position, MPI_PACKED, rank - Px, rank, MPI_COMM_WORLD, &requests[request_count++]);
            }

            // Receive from top neighbor
            if (has_top) {
                position = 0;
                MPI_Recv(top_buff_recv, 2*N_side*(sizeof(double)), MPI_PACKED, rank - Px, rank - Px, MPI_COMM_WORLD, &status[status_count++]);
                
                MPI_Wait(&requests[request_count-1], &status[status_count-1]);   // Waiting for the communication to complete
                for(int i = 0; i < N_side; i++){
                    MPI_Unpack(top_buff_recv, 2*N_side*(sizeof(double)), &position, top+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                }
                for(int i = 0; i < N_side; i++){
                    MPI_Unpack(top_buff_recv, 2*N_side*(sizeof(double)), &position, top+N_side+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                }
            }

            // Pack and send to bottom neighbor
            if (has_bottom) {
                position = 0;
                for(int i = 0; i < N_side; i++){
                    MPI_Pack(&data[N_side-1][i], 1, MPI_DOUBLE, bottom_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                }
                for(int i = 0; i < N_side; i++){
                    MPI_Pack(&data[N_side-2][i], 1, MPI_DOUBLE, bottom_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                }
                MPI_Isend(bottom_buff_send, position, MPI_PACKED, rank + Px, rank, MPI_COMM_WORLD, &requests[request_count++]);
            }

            // Receive from bottom neighbor
            if (has_bottom) {
                position = 0;
                MPI_Recv(bottom_buff_recv, 2*N_side*(sizeof(double)), MPI_PACKED, rank + Px, rank + Px, MPI_COMM_WORLD, &status[status_count++]);
    
                MPI_Wait(&requests[request_count-1], &status[status_count-1]);    // Waiting for the communication to complete
                for(int i = 0; i < N_side; i++){
                    MPI_Unpack(bottom_buff_recv, 2*N_side*(sizeof(double)), &position, bottom+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                }
                for(int i = 0; i < N_side; i++){
                    MPI_Unpack(bottom_buff_recv, 2*N_side*(sizeof(double)), &position, bottom+N_side+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                }
            }

            // Pack and send to left neighbor
            if (has_left) {
                position = 0;
                for(int i = 0; i < N_side; i++){
                    MPI_Pack(&data[i][0], 1, MPI_DOUBLE, left_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                }
                for(int i = 0; i < N_side; i++){
                    MPI_Pack(&data[i][1], 1, MPI_DOUBLE, left_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                }
                MPI_Isend(left_buff_send, position, MPI_PACKED, rank - 1, rank, MPI_COMM_WORLD, &requests[request_count++]);
            }

            // Receive from left neighbor
            if (has_left) {
                position = 0;
                MPI_Recv(left_buff_recv, 2*N_side*(sizeof(double)), MPI_PACKED, rank - 1, rank - 1, MPI_COMM_WORLD, &status[status_count++]);

                MPI_Wait(&requests[request_count-1], &status[status_count-1]);    // Waiting for the communication to complete
                for(int i = 0; i < N_side; i++){
                    MPI_Unpack(left_buff_recv, 2*N_side*(sizeof(double)), &position, left+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                }
                for(int i = 0; i < N_side; i++){
                    MPI_Unpack(left_buff_recv, 2*N_side*(sizeof(double)), &position, left+N_side+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                }
            }

            // Pack and send to right neighbor
            if (has_right) {
                position = 0;
                for(int i = 0; i < N_side; i++){
                    MPI_Pack(&data[i][N_side-1], 1, MPI_DOUBLE, right_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                }
                for(int i = 0; i < N_side; i++){
                    MPI_Pack(&data[i][N_side-2], 1, MPI_DOUBLE, right_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                }
                MPI_Isend(right_buff_send, position, MPI_PACKED, rank + 1, rank, MPI_COMM_WORLD, &requests[request_count++]);
            }

            // Receive from right neighbor
            if (has_right) {
                position = 0;
                MPI_Recv(right_buff_recv, 2*N_side*(sizeof(double)), MPI_PACKED, rank + 1, rank + 1, MPI_COMM_WORLD, &status[status_count++]);

                MPI_Wait(&requests[request_count-1], &status[status_count-1]);    // Waiting for the communication to complete
                for(int i = 0; i < N_side; i++){
                    MPI_Unpack(right_buff_recv, 2*N_side*(sizeof(double)), &position, right+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                }
                for(int i = 0; i < N_side; i++){
                    MPI_Unpack(right_buff_recv, 2*N_side*(sizeof(double)), &position, right+N_side+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                }
            }
            // Update the centre data points
            for (int i = 2; i < N_side - 2; i++) {
                for (int j = 2; j < N_side - 2; j++) {
                    data_new[i][j] = (data[i][j] + data[i-1][j] + data[i+1][j] + data[i][j-1] + data[i][j+1] + data[i][j-2] + data[i][j+2] + data[i-2][j] + data[i+2][j]) / 9.0;
                }
            }

            // Update the halo points
            if (has_top) {
                for (int i = 2; i < N_side - 2; i++) {
                    data_new[0][i] = (data[0][i] + top[i] + data[1][i] + data[0][i-1] + data[0][i+1] + top[N_side+i] + data[2][i] + data[0][i+2] + data[0][i-2]) / 9.0;
                    data_new[1][i] = (data[1][i] + top[i] + data[0][i] + data[2][i] + data[3][i] + data[1][i+1] + data[1][i+2] + data[1][i-1] + data[1][i-2]) / 9.0;
                }
            }

            if (has_bottom) {
                for (int i = 2; i < N_side - 2; i++) {
                    data_new[N_side-1][i] = (data[N_side-1][i] + bottom[i] + data[N_side-2][i] + data[N_side-3][i] + data[N_side-1][i+1] + bottom[N_side+i] + data[N_side-1][i+2] + data[N_side-1][i-2] + data[N_side-1][i-1]) / 9.0;
                    data_new[N_side-2][i] = (data[N_side-2][i] + bottom[i] + data[N_side-1][i] + data[N_side-3][i] + data[N_side-4][i] + data[N_side-2][i+1] + data[N_side-2][i+2] + data[N_side-2][i-1] + data[N_side-2][i-2]) / 9.0;
                }
            }

            if (has_left) {
                for (int i = 2; i < N_side - 2; i++) {
                    data_new[i][0] = (data[i][0] + left[i] + data[i+1][0] + data[i-1][0] + data[i][1] + left[N_side+i] + data[i+2][0] + data[i-2][0] + data[i][2]) / 9.0;
                    data_new[i][1] = (data[i][1] + left[i] + data[i][0] + data[i+1][1] + data[i-1][1] + data[i+2][1] + data[i-2][1] + data[i][2] + data[i][3]) / 9.0;
                }
            }

            if (has_right) {
                for (int i = 2; i < N_side - 2; i++) {
                    data_new[i][N_side-1] = (data[i][N_side-1] + right[i] + data[i+1][N_side-1] + data[i-1][N_side-1] + data[i][N_side-2] + right[N_side+i] + data[i+2][N_side-1] + data[i-2][N_side-1] + data[i][N_side-3]) / 9.0;
                    data_new[i][N_side-2] = (data[i][N_side-2] + right[i] + data[i][N_side-1] + data[i+1][N_side-2] + data[i-1][N_side-2] + data[i+2][N_side-2] + data[i-2][N_side-2] + data[i][N_side-3] + data[i][N_side-4]) / 9.0;
                }
            }

            // Update the inner corner points
            if (has_top && has_left) {
                data_new[0][0] = (data[0][0] + top[0] + data[1][0] + left[0] + data[0][1] + top[N_side] + data[2][0] + data[0][2] + left[N_side]) / 9.0;
                data_new[1][0] = (data[1][0] + top[0] + data[0][0] + data[2][0] + data[3][0] + data[1][1] + data[1][2] + left[1] + left[N_side+1]) / 9.0;
                data_new[0][1] = (data[0][1] + top[1] + data[1][1] + data[0][0] + data[0][2] + top[N_side+1] + data[2][1] + data[0][3] + left[0]) / 9.0;
                data_new[1][1] = (data[1][1] + top[1] + data[0][1] + data[2][1] + data[1][0] + top[1] + data[1][2] + data[1][3] + data[3][1]) / 9.0;
            }

            if (has_top && has_right) {
                data_new[0][N_side-1] = (data[0][N_side-1] + top[N_side-1] + data[1][N_side-1] + data[0][N_side-2] + right[0] + top[2*N_side-1] + data[2][N_side-1] + data[0][N_side-3] + right[N_side]) / 9.0;
                data_new[1][N_side-1] = (data[1][N_side-1] + top[N_side-1] + data[0][N_side-1] + data[2][N_side-1] + data[3][N_side-1] + data[1][N_side-2] + data[1][N_side-3] + right[N_side+1] + right[1]) / 9.0;
                data_new[0][N_side-2] = (data[0][N_side-2] + top[N_side-2] + data[1][N_side-2] + data[0][N_side-3] + data[0][N_side-1] + top[2*N_side-2] + data[2][N_side-2] + data[0][N_side-4] + right[0]) / 9.0;
                data_new[1][N_side-2] = (data[1][N_side-2] + top[N_side-2] + data[0][N_side-2] + data[2][N_side-2] + data[1][N_side-1] + data[1][N_side-3] + data[1][N_side-4] + data[3][N_side-2] + right[1]) / 9.0;
            }

            if (has_bottom && has_left) {
                data_new[N_side-1][0] = (data[N_side-1][0] + bottom[0] + data[N_side-2][0] + left[N_side-1] + data[N_side-1][1] + bottom[N_side] + data[N_side-3][0] + data[N_side-1][2] + left[2*N_side-1]) / 9.0;
                data_new[N_side-2][0] = (data[N_side-2][0] + bottom[0] + data[N_side-1][0] + data[N_side-3][0] + data[N_side-4][0] + data[N_side-2][1] + data[N_side-2][2] + left[N_side-2] + left[2*N_side-2]) / 9.0;
                data_new[N_side-1][1] = (data[N_side-1][1] + bottom[1] + data[N_side-2][1] + left[N_side-1] + data[N_side-1][0] + bottom[N_side+1] + data[N_side-3][1] + data[N_side-1][2] + data[N_side-1][3]) / 9.0;
                data_new[N_side-2][1] = (data[N_side-2][1] + bottom[1] + data[N_side-1][1] + data[N_side-3][1] + data[N_side-4][1] + data[N_side-2][0] + data[N_side-2][2] + data[N_side-2][3] + left[N_side-2]) / 9.0;
            }

            if (has_bottom && has_right) {
                data_new[N_side-1][N_side-1] = (data[N_side-1][N_side-1] + bottom[N_side-1] + data[N_side-2][N_side-1] + right[N_side-1] + data[N_side-1][N_side-2] + bottom[2*N_side-1] + data[N_side-3][N_side-1] + data[N_side-1][N_side-3] + right[2*N_side-1]) / 9.0;
                data_new[N_side-2][N_side-1] = (data[N_side-2][N_side-1] + bottom[N_side-1] + data[N_side-1][N_side-1] + data[N_side-3][N_side-1] + data[N_side-4][N_side-1] + data[N_side-2][N_side-2] + data[N_side-2][N_side-3] + right[N_side-2] + right[2*N_side-2]) / 9.0;
                data_new[N_side-1][N_side-2] = (data[N_side-1][N_side-2] + bottom[N_side-2] + data[N_side-2][N_side-2] + right[N_side-1] + data[N_side-1][N_side-3] + bottom[2*N_side-2] + data[N_side-3][N_side-2] + data[N_side-1][N_side-4] + data[N_side-1][N_side-1]) / 9.0;
                data_new[N_side-2][N_side-2] = (data[N_side-2][N_side-2] + bottom[N_side-2] + data[N_side-1][N_side-2] + data[N_side-3][N_side-2] + data[N_side-4][N_side-2] + data[N_side-2][N_side-1] + data[N_side-2][N_side-3] + data[N_side-2][N_side-4] + right[N_side-2]) / 9.0;
            }

            // Update the edge points
            if(!has_top){
                for(int i = 2; i < N_side - 2; i++){
                    data_new[0][i] = (data[0][i] + data[1][i] + data[0][i-1] + data[0][i+1] + data[0][i-2] + data[0][i+2] + data[2][i]) / 7.0;
                    data_new[1][i] = (data[1][i] + data[0][i] + data[2][i] + data[1][i-1] + data[1][i+1] + data[1][i-2] + data[1][i+2] + data[3][i]) / 8.0;
                }
            }

            if(!has_bottom){
                for(int i = 2; i < N_side - 2; i++){
                    data_new[N_side-1][i] = (data[N_side-1][i] + data[N_side-2][i] + data[N_side-1][i-1] + data[N_side-1][i+1] + data[N_side-1][i-2] + data[N_side-1][i+2] + data[N_side-3][i]) / 7.0;
                    data_new[N_side-2][i] = (data[N_side-2][i] + data[N_side-1][i] + data[N_side-3][i] + data[N_side-2][i-1] + data[N_side-2][i+1] + data[N_side-2][i-2] + data[N_side-2][i+2] + data[N_side-4][i]) / 8.0;
                }
            }

            if(!has_left){
                for(int i = 2; i < N_side - 2; i++){
                    data_new[i][0] = (data[i][0] + data[i+1][0] + data[i-1][0] + data[i][1] + data[i+2][0] + data[i-2][0] + data[i][2]) / 7.0;
                    data_new[i][1] = (data[i][1] + data[i][0] + data[i+1][1] + data[i-1][1] + data[i+2][1] + data[i-2][1] + data[i][2] + data[i][3]) / 8.0;
                }
            }

            if(!has_right){
                for(int i = 2; i < N_side - 2; i++){
                    data_new[i][N_side-1] = (data[i][N_side-1] + data[i+1][N_side-1] + data[i-1][N_side-1] + data[i][N_side-2] + data[i+2][N_side-1] + data[i-2][N_side-1] + data[i][N_side-3]) / 7.0;
                    data_new[i][N_side-2] = (data[i][N_side-2] + data[i][N_side-1] + data[i+1][N_side-2] + data[i-1][N_side-2] + data[i+2][N_side-2] + data[i-2][N_side-2] + data[i][N_side-3] + data[i][N_side-4]) / 8.0;
                }
            }

            // Update the outer corner points
            if(!has_top && !has_left){
                data_new[0][0] = (data[0][0] + data[1][0] + data[0][1] + data[0][2] + data[2][0]) / 5.0;
                data_new[1][0] = (data[1][0] + data[0][0] + data[2][0] + data[1][1] + data[1][2] + data[3][0]) / 6.0;
                data_new[0][1] = (data[0][1] + data[1][1] + data[0][0] + data[0][2] + data[2][1] + data[0][3]) / 6.0;
                data_new[1][1] = (data[1][1] + data[0][1] + data[2][1] + data[1][0] + data[1][2] + data[3][1] + data[1][3]) / 7.0;
            }

            if(!has_top && !has_right){
                data_new[0][N_side-1] = (data[0][N_side-1] + data[1][N_side-1] + data[0][N_side-2] + data[0][N_side-3] + data[2][N_side-1]) / 5.0;
                data_new[1][N_side-1] = (data[1][N_side-1] + data[0][N_side-1] + data[2][N_side-1] + data[1][N_side-2] + data[1][N_side-3] + data[3][N_side-1]) / 6.0;
                data_new[0][N_side-2] = (data[0][N_side-2] + data[1][N_side-2] + data[0][N_side-1] + data[0][N_side-3] + data[2][N_side-2] + data[0][N_side-4]) / 6.0;
                data_new[1][N_side-2] = (data[1][N_side-2] + data[0][N_side-2] + data[2][N_side-2] + data[1][N_side-1] + data[1][N_side-3] + data[3][N_side-2] + data[1][N_side-4]) / 7.0;
            }

            if(!has_bottom && !has_left){
                data_new[N_side-1][0] = (data[N_side-1][0] + data[N_side-2][0] + data[N_side-1][1] + data[N_side-1][2] + data[N_side-3][0]) / 5.0;
                data_new[N_side-2][0] = (data[N_side-2][0] + data[N_side-1][0] + data[N_side-3][0] + data[N_side-2][1] + data[N_side-2][2] + data[N_side-4][0]) / 6.0;
                data_new[N_side-1][1] = (data[N_side-1][1] + data[N_side-2][1] + data[N_side-1][0] + data[N_side-1][2] + data[N_side-3][1] + data[N_side-1][3]) / 6.0;
                data_new[N_side-2][1] = (data[N_side-2][1] + data[N_side-1][1] + data[N_side-3][1] + data[N_side-2][0] + data[N_side-2][2] + data[N_side-4][1] + data[N_side-2][3]) / 7.0;
            }

            if(!has_bottom && !has_right){
                data_new[N_side-1][N_side-1] = (data[N_side-1][N_side-1] + data[N_side-2][N_side-1] + data[N_side-1][N_side-2] + data[N_side-1][N_side-3] + data[N_side-3][N_side-1]) / 5.0;
                data_new[N_side-2][N_side-1] = (data[N_side-2][N_side-1] + data[N_side-1][N_side-1] + data[N_side-3][N_side-1] + data[N_side-2][N_side-2] + data[N_side-2][N_side-3] + data[N_side-4][N_side-1]) / 6.0;
                data_new[N_side-1][N_side-2] = (data[N_side-1][N_side-2] + data[N_side-2][N_side-2] + data[N_side-1][N_side-1] + data[N_side-1][N_side-3] + data[N_side-3][N_side-2] + data[N_side-1][N_side-4]) / 6.0;
                data_new[N_side-2][N_side-2] = (data[N_side-2][N_side-2] + data[N_side-1][N_side-2] + data[N_side-3][N_side-2] + data[N_side-2][N_side-1] + data[N_side-2][N_side-3] + data[N_side-4][N_side-2] + data[N_side-2][N_side-4]) / 7.0;
            }

            if(has_top && !has_left){
                data_new[0][0] = (data[0][0] + top[0] + data[1][0] + data[0][1] + top[N_side] + data[2][0] + data[0][2]) / 7.0;
                data_new[1][0] = (data[1][0] + top[0] + data[0][0] + data[2][0] + data[1][1] + data[1][2] + data[3][0]) / 7.0;
                data_new[0][1] = (data[0][1] + top[1] + data[1][1] + data[0][0] + data[0][2] + top[N_side+1] + data[2][1] + data[0][3]) / 8.0;
                data_new[1][1] = (data[1][1] + top[1] + data[0][1] + data[2][1] + data[1][0] + data[1][2] + data[1][3] + data[3][1]) / 8.0;
            }

            if(has_top && !has_right){
                data_new[0][N_side-1] = (data[0][N_side-1] + top[N_side-1] + data[1][N_side-1] + data[0][N_side-2] + top[2*N_side-1] + data[2][N_side-1] + data[0][N_side-3]) / 7.0;
                data_new[1][N_side-1] = (data[1][N_side-1] + top[N_side-1] + data[0][N_side-1] + data[2][N_side-1] + data[1][N_side-2] + data[1][N_side-3] + data[3][N_side-1]) / 7.0;
                data_new[0][N_side-2] = (data[0][N_side-2] + top[N_side-2] + data[1][N_side-2] + data[0][N_side-1] + top[2*N_side-2] + data[2][N_side-2] + data[0][N_side-3] + data[0][N_side-4]) / 8.0;
                data_new[1][N_side-2] = (data[1][N_side-2] + top[N_side-2] + data[0][N_side-2] + data[2][N_side-2] + data[1][N_side-1] + data[1][N_side-3] + data[1][N_side-4] + data[3][N_side-2]) / 8.0;
            }

            if(has_bottom && !has_left){
                data_new[N_side-1][0] = (data[N_side-1][0] + bottom[0] + data[N_side-2][0] + bottom[N_side] + data[N_side-1][1] + data[N_side-3][0] + data[N_side-1][2]) / 7.0;
                data_new[N_side-2][0] = (data[N_side-2][0] + bottom[0] + data[N_side-1][0] + data[N_side-3][0] + data[N_side-2][1] + data[N_side-2][2] + data[N_side-4][0]) / 7.0;
                data_new[N_side-1][1] = (data[N_side-1][1] + bottom[1] + data[N_side-2][1] + bottom[N_side+1] + data[N_side-1][0] + data[N_side-3][1] + data[N_side-1][2] + data[N_side-1][3]) / 8.0;
                data_new[N_side-2][1] = (data[N_side-2][1] + bottom[1] + data[N_side-1][1] + data[N_side-3][1] + data[N_side-2][0] + data[N_side-2][2] + data[N_side-4][1] + data[N_side-2][3]) / 8.0;
            }

            if(has_bottom && !has_right){
                data_new[N_side-1][N_side-1] = (data[N_side-1][N_side-1] + bottom[N_side-1] + data[N_side-2][N_side-1] + bottom[2*N_side-1] + data[N_side-1][N_side-2] + data[N_side-3][N_side-1] + data[N_side-1][N_side-3]) / 7.0;
                data_new[N_side-2][N_side-1] = (data[N_side-2][N_side-1] + bottom[N_side-1] + data[N_side-1][N_side-1] + data[N_side-3][N_side-1] + data[N_side-2][N_side-2] + data[N_side-2][N_side-3] + data[N_side-4][N_side-1]) / 7.0;
                data_new[N_side-1][N_side-2] = (data[N_side-1][N_side-2] + bottom[N_side-2] + data[N_side-2][N_side-2] + bottom[2*N_side-2] + data[N_side-1][N_side-1] + data[N_side-3][N_side-2] + data[N_side-1][N_side-3] + data[N_side-1][N_side-4]) / 8.0;
                data_new[N_side-2][N_side-2] = (data[N_side-2][N_side-2] + bottom[N_side-2] + data[N_side-1][N_side-2] + data[N_side-3][N_side-2] + data[N_side-2][N_side-1] + data[N_side-2][N_side-3] + data[N_side-4][N_side-2] + data[N_side-2][N_side-4]) / 8.0;
            }

            if(has_right && !has_top){
                data_new[0][N_side-1] = (data[0][N_side-1] + data[1][N_side-1] + data[0][N_side-2] + data[0][N_side-3] + data[2][N_side-1] + right[0] + right[N_side]) / 7.0;
                data_new[1][N_side-1] = (data[1][N_side-1] + data[0][N_side-1] + data[2][N_side-1] + data[1][N_side-2] + data[1][N_side-3] + data[3][N_side-1] + right[1] + right[N_side+1]) / 8.0;
                data_new[0][N_side-2] = (data[0][N_side-2] + data[1][N_side-2] + data[0][N_side-1] + data[0][N_side-3] + data[2][N_side-2] + right[0] + data[0][N_side-4]) / 7.0;
                data_new[1][N_side-2] = (data[1][N_side-2] + data[0][N_side-2] + data[2][N_side-2] + data[1][N_side-1] + data[1][N_side-3] + data[3][N_side-2] + right[1] + data[1][N_side-4]) / 8.0;
            }

            if(has_right && !has_bottom){
                data_new[N_side-1][N_side-1] = (data[N_side-1][N_side-1] + data[N_side-2][N_side-1] + data[N_side-1][N_side-2] + data[N_side-1][N_side-3] + data[N_side-3][N_side-1] + right[N_side-1] + right[2*N_side-1]) / 7.0;
                data_new[N_side-2][N_side-1] = (data[N_side-2][N_side-1] + data[N_side-1][N_side-1] + data[N_side-3][N_side-1] + data[N_side-2][N_side-2] + data[N_side-2][N_side-3] + data[N_side-4][N_side-1] + right[N_side-2] + right[2*N_side-2]) / 8.0;
                data_new[N_side-1][N_side-2] = (data[N_side-1][N_side-2] + data[N_side-2][N_side-2] + data[N_side-1][N_side-1] + data[N_side-1][N_side-3] + data[N_side-3][N_side-2] + right[N_side-1] + data[N_side-1][N_side-4]) / 7.0;
                data_new[N_side-2][N_side-2] = (data[N_side-2][N_side-2] + data[N_side-1][N_side-2] + data[N_side-3][N_side-2] + data[N_side-2][N_side-1] + data[N_side-2][N_side-3] + data[N_side-4][N_side-2] + right[N_side-2] + data[N_side-2][N_side-4]) / 8.0;
            }

            if(has_left && !has_top){
                data_new[0][0] = (data[0][0] + data[1][0] + data[0][1] + data[0][2] + data[2][0] + left[0] + left[N_side]) / 7.0;
                data_new[1][0] = (data[1][0] + data[0][0] + data[2][0] + data[1][1] + data[1][2] + data[3][0] + left[1] + left[N_side+1]) / 8.0;
                data_new[0][1] = (data[0][1] + data[1][1] + data[0][0] + data[0][2] + data[2][1] + data[0][3] + left[0]) / 7.0;
                data_new[1][1] = (data[1][1] + data[0][1] + data[2][1] + data[1][0] + data[1][2] + data[3][1] + left[1] + data[1][3]) / 8.0;
            }

            if(has_left && !has_bottom){
                data_new[N_side-1][0] = (data[N_side-1][0] + data[N_side-2][0] + data[N_side-1][1] + data[N_side-1][2] + data[N_side-3][0] + left[N_side-1] + left[2*N_side-1]) / 7.0;
                data_new[N_side-2][0] = (data[N_side-2][0] + data[N_side-1][0] + data[N_side-3][0] + data[N_side-2][1] + data[N_side-2][2] + data[N_side-4][0] + left[N_side-2] + left[2*N_side-2]) / 8.0;
                data_new[N_side-1][1] = (data[N_side-1][1] + data[N_side-2][1] + data[N_side-1][0] + data[N_side-1][2] + data[N_side-3][1] + data[N_side-1][3] + left[N_side-1]) / 7.0;
                data_new[N_side-2][1] = (data[N_side-2][1] + data[N_side-1][1] + data[N_side-3][1] + data[N_side-2][0] + data[N_side-2][2] + data[N_side-4][1] + left[N_side-2] + data[N_side-2][3]) / 8.0;
            }

            data = data_new;

            // Reset request and status count for next iteration
            request_count = 0;
            status_count = 0;
        }
    }
    
    eTime = MPI_Wtime();    // End timer

    double time = eTime - sTime;    
    double max_time;
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);    // Get the maximum time taken by any process
    
    write_matrix_to_csv(data, N_side, rank);

    if (rank == 0) {
        printf("Time taken for computing %d-point stencil: %f\n", stencil, max_time);
    }
    
    for (int i = 0; i < N_side; i++){
        free(data[i]);
        // free(data_new[i]);
    }

    free(data);
    // free(data_new);
    MPI_Finalize();
    return 0;
}