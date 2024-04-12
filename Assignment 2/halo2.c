#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

void write_matrix_to_csv(double **data, int N_side, int process_rank) {
    char filename[256];
    sprintf(filename, "rank2_%d.csv", process_rank);

    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }

    for (int i = 0; i < N_side; i++) {
        for (int j = 0; j < N_side; j++) {
            fprintf(file, "%f", data[i][j]);
            if (j < N_side - 1)
                fprintf(file, ", ");
        }
        fprintf(file, "\n");
    }

    fclose(file);
}



int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5) {
        if (rank == 0) {
            printf("Usage: %s Px N_side num_time_steps seed\n", argv[0]);
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

    srand(seed * (rank + 10));
    // Creating the data matrix
    double** data = (double**)malloc(N_side * sizeof(double*));
    double** data2 = (double**)malloc(N_side * sizeof(double*));
    double** data_new = (double**)malloc(N_side * sizeof(double*));
    double** data2_new = (double**)malloc(N_side * sizeof(double*));
    
    for (double i = 0; i < N_side; i++){
        data[(int)i] = (double*)malloc(N_side * sizeof(double));
        data2[(int)i] = (double*)malloc(N_side * sizeof(double));
        data_new[(int)i] = (double*)malloc(N_side * sizeof(double));
        data2_new[(int)i] = (double*)malloc(N_side * sizeof(double));
    }
    for (double i = 0; i < N_side; i++) {
        for (double j = 0; j < N_side; j++) {
            // data[(int)i][(int)j] = abs((rand() + (i * rand()) + (j * rank)) / 100.0);
            data[(int)i][(int)j] = (i+j);
            // *(rank+1);
            data2[(int)i][(int)j] = data[(int)i][(int)j];
            // if(rank == 0)printf("%f ",data2[(int)i][(int)j]);
        }
        // if(rank == 0)printf("\n");
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

    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, y, rank, &new_comm);

    int new_rank;
    MPI_Comm_rank(new_comm, &new_rank);
    
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
        double top_buff_send[2*N_side], top_buff_recv[2*N_side*Px];
        double bottom_buff_send[2*N_side], bottom_buff_recv[2*N_side*Px];
        double left_buff_send[2*N_side], left_buff_recv[2*N_side];
        double right_buff_send[2*N_side], right_buff_recv[2*N_side];
        double top[2*N_side], bottom[2*N_side], left[2*N_side], right[2*N_side];
        double top_all[2*N_side*(Px)], bottom_all[2*N_side*(Px)];
        int position;

        if(has_top && has_left){
            position = 0;
            for(int i = 0; i < N_side; i++){
                MPI_Pack(&data2[0][i], 1, MPI_DOUBLE, top_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                // top_buff_send[i] = data2[0][i];
            }
            for(int i = 0; i < N_side; i++){
                MPI_Pack(&data2[1][i], 1, MPI_DOUBLE, top_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                // top_buff_send[N_side+i] = data2[1][i];
            }
            // MPI_Isend(top_buff_send, position, MPI_PACKED, y*Px, rank, MPI_COMM_WORLD, &requests[request_count++]);
        }
        
        if(has_top && !has_left){
            position = 0;
            for(int i = 0; i < N_side; i++){
                MPI_Pack(&data2[0][i], 1, MPI_DOUBLE, top_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
            }
            for(int i = 0; i < N_side; i++){
                MPI_Pack(&data2[1][i], 1, MPI_DOUBLE, top_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
            }
        }
        // printf("Rank: %d\n", position);
        // if(new_rank == 1 && has_top){
        //     for(int i = 0; i < 2*N_side; i++){
        //         printf("%f ", top_buff_send[i]);
        //     }
        //     printf("\n");
        // }
        if(has_top) {
            MPI_Gather(top_buff_send, 2*N_side, MPI_DOUBLE, top_buff_recv, 2*N_side, MPI_DOUBLE, 0, new_comm);
            // MPI_Barrier(new_comm);
        }
        // printf("Rank: %d\n", rank);
        
        // position = 0;
        // double buffer[2*N_side*(Px)];
        // for(int i = 0; i < 2*N_side*(Px); i++){
        //     MPI_Unpack(top_buff_recv, 2*N_side*(Px)*(sizeof(double)), &position, buffer+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        // }
        // if(new_rank == 0 && has_top){
        //     for(int i = 0 ; i<Px ; i++){
        //         for(int j = 0; j<2*N_side; j++){
        //             printf("%f ", buffer[i*2*N_side+j]);
        //         }
        //         printf("\n");
        //     }
        // }

        if(has_top && !has_left){
            
            MPI_Isend(top_buff_recv, Px*position, MPI_PACKED, rank - Px, rank, MPI_COMM_WORLD, &requests[request_count++]);
            
        }

        if(has_bottom && !has_left){
            position = 0;
            // printf("Rank: %d\n", rank);
            MPI_Recv(top_buff_recv, 2*N_side*(Px), MPI_DOUBLE, rank + Px, rank + Px, MPI_COMM_WORLD, &status[status_count++]);
            // MPI_Wait(&requests[request_count-1], &status[status_count-1]);   // Waiting for the communication to complete
            for(int i = 0; i < 2*N_side*(Px); i++){
                MPI_Unpack(top_buff_recv, 2*N_side*(Px)*(sizeof(double)), &position, bottom_all+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
            // for(int i = 0; i < Px; i++){
            //     for(int j = 0; j < 2*N_side; j++){
            //         printf("%f ", bottom_all[i*2*N_side+j]);
            //     }
            //     printf("\n");
            // }
               
        }
        if(has_bottom) MPI_Scatter(bottom_all, 2*N_side, MPI_DOUBLE, bottom, 2*N_side, MPI_DOUBLE, 0, new_comm);
        // if(rank==1){
        //     for(int i = 0; i < 2*N_side; i++){
        //         printf("%f ", bottom[i]);
        //     }
        // }

        if(has_bottom && has_left){
            position = 0;
            for(int i = 0; i < N_side; i++){
                MPI_Pack(&data2[N_side-1][i], 1, MPI_DOUBLE, bottom_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                // top_buff_send[i] = data2[0][i];
            }
            for(int i = 0; i < N_side; i++){
                MPI_Pack(&data2[N_side-2][i], 1, MPI_DOUBLE, bottom_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
                // top_buff_send[N_side+i] = data2[1][i];
            }
        }

        if(has_bottom && !has_left){
            position = 0;
            for(int i = 0; i < N_side; i++){
                MPI_Pack(&data2[N_side-1][i], 1, MPI_DOUBLE, bottom_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
            }
            for(int i = 0; i < N_side; i++){
                MPI_Pack(&data2[N_side-2][i], 1, MPI_DOUBLE, bottom_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
            }
        }

        if(has_bottom) {
            MPI_Gather(bottom_buff_send, 2*N_side, MPI_DOUBLE, bottom_buff_recv, 2*N_side, MPI_DOUBLE, 0, new_comm);
            // MPI_Barrier(new_comm);
        }

        if(has_bottom && !has_left){
            MPI_Isend(bottom_buff_recv, Px*position, MPI_PACKED, rank + Px, rank, MPI_COMM_WORLD, &requests[request_count++]);  
        }

        if(has_top && !has_left){
            position = 0;
            // printf("Rank: %d\n", rank);
            MPI_Recv(bottom_buff_recv, 2*N_side*(Px), MPI_DOUBLE, rank - Px, rank - Px, MPI_COMM_WORLD, &status[status_count++]);
            // MPI_Wait(&requests[request_count-1], &status[status_count-1]);   // Waiting for the communication to complete
            for(int i = 0; i < 2*N_side*(Px); i++){
                MPI_Unpack(bottom_buff_recv, 2*N_side*(Px)*(sizeof(double)), &position, top_all+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
            // for(int i = 0; i < Px; i++){
            //     for(int j = 0; j < 2*N_side; j++){
            //         printf("%f ", bottom_all[i*2*N_side+j]);
            //     }
            //     printf("\n");
            // }
               
        }

        if(has_top) MPI_Scatter(top_all, 2*N_side, MPI_DOUBLE, top, 2*N_side, MPI_DOUBLE, 0, new_comm);
        if(rank == 5){
            for(int i = 0; i < 2*N_side; i++){
                printf("%f ", top[i]);
            }
            printf("\n");
        }

        // Pack and send to left neighbor
        if (has_left) {
            position = 0;
            for(int i = 0; i < N_side; i++){
                MPI_Pack(&data2[i][0], 1, MPI_DOUBLE, left_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
            }
            for(int i = 0; i < N_side; i++){
                MPI_Pack(&data2[i][1], 1, MPI_DOUBLE, left_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
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
                MPI_Pack(&data2[i][N_side-1], 1, MPI_DOUBLE, right_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
            }
            for(int i = 0; i < N_side; i++){
                MPI_Pack(&data2[i][N_side-2], 1, MPI_DOUBLE, right_buff_send, N_side*(sizeof(double)), &position, MPI_COMM_WORLD);
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
        // Update the centre data2 points
        for (int i = 2; i < N_side - 2; i++) {
            for (int j = 2; j < N_side - 2; j++) {
                data2_new[i][j] = (data2[i][j] + data2[i-1][j] + data2[i+1][j] + data2[i][j-1] + data2[i][j+1] + data2[i][j-2] + data2[i][j+2] + data2[i-2][j] + data2[i+2][j]) / 9.0;
            }
        }

        // Update the halo points
        if (has_top) {
            for (int i = 2; i < N_side - 2; i++) {
                data_new[0][i] = (data2[0][i] + top[i] + data2[1][i] + data2[0][i-1] + data2[0][i+1] + top[N_side+i] + data2[2][i] + data2[0][i+2] + data2[0][i-2]) / 9.0;
                data2_new[1][i] = (data2[1][i] + top[i] + data2[0][i] + data2[2][i] + data2[3][i] + data2[1][i+1] + data2[1][i+2] + data2[1][i-1] + data2[1][i-2]) / 9.0;
            }
        }

        if (has_bottom) {
            for (int i = 2; i < N_side - 2; i++) {
                data2_new[N_side-1][i] = (data2[N_side-1][i] + bottom[i] + data2[N_side-2][i] + data2[N_side-3][i] + data2[N_side-1][i+1] + bottom[N_side+i] + data2[N_side-1][i+2] + data2[N_side-1][i-2] + data2[N_side-1][i-1]) / 9.0;
                data2_new[N_side-2][i] = (data2[N_side-2][i] + bottom[i] + data2[N_side-1][i] + data2[N_side-3][i] + data2[N_side-4][i] + data2[N_side-2][i+1] + data2[N_side-2][i+2] + data2[N_side-2][i-1] + data2[N_side-2][i-2]) / 9.0;
            }
        }

        if (has_left) {
            for (int i = 2; i < N_side - 2; i++) {
                data2_new[i][0] = (data2[i][0] + left[i] + data2[i+1][0] + data2[i-1][0] + data2[i][1] + left[N_side+i] + data2[i+2][0] + data2[i-2][0] + data2[i][2]) / 9.0;
                data2_new[i][1] = (data2[i][1] + left[i] + data2[i][0] + data2[i+1][1] + data2[i-1][1] + data2[i+2][1] + data2[i-2][1] + data2[i][2] + data2[i][3]) / 9.0;
            }
        }

        if (has_right) {
            for (int i = 2; i < N_side - 2; i++) {
                data2_new[i][N_side-1] = (data2[i][N_side-1] + right[i] + data2[i+1][N_side-1] + data2[i-1][N_side-1] + data2[i][N_side-2] + right[N_side+i] + data2[i+2][N_side-1] + data2[i-2][N_side-1] + data2[i][N_side-3]) / 9.0;
                data2_new[i][N_side-2] = (data2[i][N_side-2] + right[i] + data2[i][N_side-1] + data2[i+1][N_side-2] + data2[i-1][N_side-2] + data2[i+2][N_side-2] + data2[i-2][N_side-2] + data2[i][N_side-3] + data2[i][N_side-4]) / 9.0;
            }
        }

        // Update the inner corner points
        if (has_top && has_left) {
            data2_new[0][0] = (data2[0][0] + top[0] + data2[1][0] + left[0] + data2[0][1] + top[N_side] + data2[2][0] + data2[0][2] + left[N_side]) / 9.0;
            data2_new[1][0] = (data2[1][0] + top[0] + data2[0][0] + data2[2][0] + data2[3][0] + data2[1][1] + data2[1][2] + left[1] + left[N_side+1]) / 9.0;
            data2_new[0][1] = (data2[0][1] + top[1] + data2[1][1] + data2[0][0] + data2[0][2] + top[N_side+1] + data2[2][1] + data2[0][3] + left[0]) / 9.0;
            data2_new[1][1] = (data2[1][1] + top[1] + data2[0][1] + data2[2][1] + data2[1][0] + top[1] + data2[1][2] + data2[1][3] + data2[3][1]) / 9.0;
        }

        if (has_top && has_right) {
            data2_new[0][N_side-1] = (data2[0][N_side-1] + top[N_side-1] + data2[1][N_side-1] + data2[0][N_side-2] + right[0] + top[2*N_side-1] + data2[2][N_side-1] + data2[0][N_side-3] + right[N_side]) / 9.0;
            data2_new[1][N_side-1] = (data2[1][N_side-1] + top[N_side-1] + data2[0][N_side-1] + data2[2][N_side-1] + data2[3][N_side-1] + data2[1][N_side-2] + data2[1][N_side-3] + right[N_side+1] + right[1]) / 9.0;
            data2_new[0][N_side-2] = (data2[0][N_side-2] + top[N_side-2] + data2[1][N_side-2] + data2[0][N_side-3] + data2[0][N_side-1] + top[2*N_side-2] + data2[2][N_side-2] + data2[0][N_side-4] + right[0]) / 9.0;
            data2_new[1][N_side-2] = (data2[1][N_side-2] + top[N_side-2] + data2[0][N_side-2] + data2[2][N_side-2] + data2[1][N_side-1] + data2[1][N_side-3] + data2[1][N_side-4] + data2[3][N_side-2] + right[1]) / 9.0;
        }

        if (has_bottom && has_left) {
            data2_new[N_side-1][0] = (data2[N_side-1][0] + bottom[0] + data2[N_side-2][0] + left[N_side-1] + data2[N_side-1][1] + bottom[N_side] + data2[N_side-3][0] + data2[N_side-1][2] + left[2*N_side-1]) / 9.0;
            data2_new[N_side-2][0] = (data2[N_side-2][0] + bottom[0] + data2[N_side-1][0] + data2[N_side-3][0] + data2[N_side-4][0] + data2[N_side-2][1] + data2[N_side-2][2] + left[N_side-2] + left[2*N_side-2]) / 9.0;
            data2_new[N_side-1][1] = (data2[N_side-1][1] + bottom[1] + data2[N_side-2][1] + left[N_side-1] + data2[N_side-1][0] + bottom[N_side+1] + data2[N_side-3][1] + data2[N_side-1][2] + data2[N_side-1][3]) / 9.0;
            data2_new[N_side-2][1] = (data2[N_side-2][1] + bottom[1] + data2[N_side-1][1] + data2[N_side-3][1] + data2[N_side-4][1] + data2[N_side-2][0] + data2[N_side-2][2] + data2[N_side-2][3] + left[N_side-2]) / 9.0;
        }

        if (has_bottom && has_right) {
            data2_new[N_side-1][N_side-1] = (data2[N_side-1][N_side-1] + bottom[N_side-1] + data2[N_side-2][N_side-1] + right[N_side-1] + data2[N_side-1][N_side-2] + bottom[2*N_side-1] + data2[N_side-3][N_side-1] + data2[N_side-1][N_side-3] + right[2*N_side-1]) / 9.0;
            data2_new[N_side-2][N_side-1] = (data2[N_side-2][N_side-1] + bottom[N_side-1] + data2[N_side-1][N_side-1] + data2[N_side-3][N_side-1] + data2[N_side-4][N_side-1] + data2[N_side-2][N_side-2] + data2[N_side-2][N_side-3] + right[N_side-2] + right[2*N_side-2]) / 9.0;
            data2_new[N_side-1][N_side-2] = (data2[N_side-1][N_side-2] + bottom[N_side-2] + data2[N_side-2][N_side-2] + right[N_side-1] + data2[N_side-1][N_side-3] + bottom[2*N_side-2] + data2[N_side-3][N_side-2] + data2[N_side-1][N_side-4] + data2[N_side-1][N_side-1]) / 9.0;
            data2_new[N_side-2][N_side-2] = (data2[N_side-2][N_side-2] + bottom[N_side-2] + data2[N_side-1][N_side-2] + data2[N_side-3][N_side-2] + data2[N_side-4][N_side-2] + data2[N_side-2][N_side-1] + data2[N_side-2][N_side-3] + data2[N_side-2][N_side-4] + right[N_side-2]) / 9.0;
        }

        // Update the edge points
        if(!has_top){
            for(int i = 2; i < N_side - 2; i++){
                data2_new[0][i] = (data2[0][i] + data2[1][i] + data2[0][i-1] + data2[0][i+1] + data2[0][i-2] + data2[0][i+2] + data2[2][i]) / 7.0;
                data2_new[1][i] = (data2[1][i] + data2[0][i] + data2[2][i] + data2[1][i-1] + data2[1][i+1] + data2[1][i-2] + data2[1][i+2] + data2[3][i]) / 8.0;
            }
        }

        if(!has_bottom){
            for(int i = 2; i < N_side - 2; i++){
                data2_new[N_side-1][i] = (data2[N_side-1][i] + data2[N_side-2][i] + data2[N_side-1][i-1] + data2[N_side-1][i+1] + data2[N_side-1][i-2] + data2[N_side-1][i+2] + data2[N_side-3][i]) / 7.0;
                data2_new[N_side-2][i] = (data2[N_side-2][i] + data2[N_side-1][i] + data2[N_side-3][i] + data2[N_side-2][i-1] + data2[N_side-2][i+1] + data2[N_side-2][i-2] + data2[N_side-2][i+2] + data2[N_side-4][i]) / 8.0;
            }
        }

        if(!has_left){
            for(int i = 2; i < N_side - 2; i++){
                data2_new[i][0] = (data2[i][0] + data2[i+1][0] + data2[i-1][0] + data2[i][1] + data2[i+2][0] + data2[i-2][0] + data2[i][2]) / 7.0;
                data2_new[i][1] = (data2[i][1] + data2[i][0] + data2[i+1][1] + data2[i-1][1] + data2[i+2][1] + data2[i-2][1] + data2[i][2] + data2[i][3]) / 8.0;
            }
        }

        if(!has_right){
            for(int i = 2; i < N_side - 2; i++){
                data2_new[i][N_side-1] = (data2[i][N_side-1] + data2[i+1][N_side-1] + data2[i-1][N_side-1] + data2[i][N_side-2] + data2[i+2][N_side-1] + data2[i-2][N_side-1] + data2[i][N_side-3]) / 7.0;
                data2_new[i][N_side-2] = (data2[i][N_side-2] + data2[i][N_side-1] + data2[i+1][N_side-2] + data2[i-1][N_side-2] + data2[i+2][N_side-2] + data2[i-2][N_side-2] + data2[i][N_side-3] + data2[i][N_side-4]) / 8.0;
            }
        }

        // Update the outer corner points
        if(!has_top && !has_left){
            data2_new[0][0] = (data2[0][0] + data2[1][0] + data2[0][1] + data2[0][2] + data2[2][0]) / 5.0;
            data2_new[1][0] = (data2[1][0] + data2[0][0] + data2[2][0] + data2[1][1] + data2[1][2] + data2[3][0]) / 6.0;
            data2_new[0][1] = (data2[0][1] + data2[1][1] + data2[0][0] + data2[0][2] + data2[2][1] + data2[0][3]) / 6.0;
            data2_new[1][1] = (data2[1][1] + data2[0][1] + data2[2][1] + data2[1][0] + data2[1][2] + data2[3][1] + data2[1][3]) / 7.0;
        }

        if(!has_top && !has_right){
            data2_new[0][N_side-1] = (data2[0][N_side-1] + data2[1][N_side-1] + data2[0][N_side-2] + data2[0][N_side-3] + data2[2][N_side-1]) / 5.0;
            data2_new[1][N_side-1] = (data2[1][N_side-1] + data2[0][N_side-1] + data2[2][N_side-1] + data2[1][N_side-2] + data2[1][N_side-3] + data2[3][N_side-1]) / 6.0;
            data2_new[0][N_side-2] = (data2[0][N_side-2] + data2[1][N_side-2] + data2[0][N_side-1] + data2[0][N_side-3] + data2[2][N_side-2] + data2[0][N_side-4]) / 6.0;
            data2_new[1][N_side-2] = (data2[1][N_side-2] + data2[0][N_side-2] + data2[2][N_side-2] + data2[1][N_side-1] + data2[1][N_side-3] + data2[3][N_side-2] + data2[1][N_side-4]) / 7.0;
        }

        if(!has_bottom && !has_left){
            data2_new[N_side-1][0] = (data2[N_side-1][0] + data2[N_side-2][0] + data2[N_side-1][1] + data2[N_side-1][2] + data2[N_side-3][0]) / 5.0;
            data2_new[N_side-2][0] = (data2[N_side-2][0] + data2[N_side-1][0] + data2[N_side-3][0] + data2[N_side-2][1] + data2[N_side-2][2] + data2[N_side-4][0]) / 6.0;
            data2_new[N_side-1][1] = (data2[N_side-1][1] + data2[N_side-2][1] + data2[N_side-1][0] + data2[N_side-1][2] + data2[N_side-3][1] + data2[N_side-1][3]) / 6.0;
            data2_new[N_side-2][1] = (data2[N_side-2][1] + data2[N_side-1][1] + data2[N_side-3][1] + data2[N_side-2][0] + data2[N_side-2][2] + data2[N_side-4][1] + data2[N_side-2][3]) / 7.0;
        }

        if(!has_bottom && !has_right){
            data2_new[N_side-1][N_side-1] = (data2[N_side-1][N_side-1] + data2[N_side-2][N_side-1] + data2[N_side-1][N_side-2] + data2[N_side-1][N_side-3] + data2[N_side-3][N_side-1]) / 5.0;
            data2_new[N_side-2][N_side-1] = (data2[N_side-2][N_side-1] + data2[N_side-1][N_side-1] + data2[N_side-3][N_side-1] + data2[N_side-2][N_side-2] + data2[N_side-2][N_side-3] + data2[N_side-4][N_side-1]) / 6.0;
            data2_new[N_side-1][N_side-2] = (data2[N_side-1][N_side-2] + data2[N_side-2][N_side-2] + data2[N_side-1][N_side-1] + data2[N_side-1][N_side-3] + data2[N_side-3][N_side-2] + data2[N_side-1][N_side-4]) / 6.0;
            data2_new[N_side-2][N_side-2] = (data2[N_side-2][N_side-2] + data2[N_side-1][N_side-2] + data2[N_side-3][N_side-2] + data2[N_side-2][N_side-1] + data2[N_side-2][N_side-3] + data2[N_side-4][N_side-2] + data2[N_side-2][N_side-4]) / 7.0;
        }

        if(has_top && !has_left){
            data2_new[0][0] = (data2[0][0] + top[0] + data2[1][0] + data2[0][1] + top[N_side] + data2[2][0] + data2[0][2]) / 7.0;
            data2_new[1][0] = (data2[1][0] + top[0] + data2[0][0] + data2[2][0] + data2[1][1] + data2[1][2] + data2[3][0]) / 7.0;
            data2_new[0][1] = (data2[0][1] + top[1] + data2[1][1] + data2[0][0] + data2[0][2] + top[N_side+1] + data2[2][1] + data2[0][3]) / 8.0;
            data2_new[1][1] = (data2[1][1] + top[1] + data2[0][1] + data2[2][1] + data2[1][0] + data2[1][2] + data2[1][3] + data2[3][1]) / 8.0;
        }

        if(has_top && !has_right){
            data2_new[0][N_side-1] = (data2[0][N_side-1] + top[N_side-1] + data2[1][N_side-1] + data2[0][N_side-2] + top[2*N_side-1] + data2[2][N_side-1] + data2[0][N_side-3]) / 7.0;
            data2_new[1][N_side-1] = (data2[1][N_side-1] + top[N_side-1] + data2[0][N_side-1] + data2[2][N_side-1] + data2[1][N_side-2] + data2[1][N_side-3] + data2[3][N_side-1]) / 7.0;
            data2_new[0][N_side-2] = (data2[0][N_side-2] + top[N_side-2] + data2[1][N_side-2] + data2[0][N_side-1] + top[2*N_side-2] + data2[2][N_side-2] + data2[0][N_side-3] + data2[0][N_side-4]) / 8.0;
            data2_new[1][N_side-2] = (data2[1][N_side-2] + top[N_side-2] + data2[0][N_side-2] + data2[2][N_side-2] + data2[1][N_side-1] + data2[1][N_side-3] + data2[1][N_side-4] + data2[3][N_side-2]) / 8.0;
        }

        if(has_bottom && !has_left){
            data2_new[N_side-1][0] = (data2[N_side-1][0] + bottom[0] + data2[N_side-2][0] + bottom[N_side] + data2[N_side-1][1] + data2[N_side-3][0] + data2[N_side-1][2]) / 7.0;
            data2_new[N_side-2][0] = (data2[N_side-2][0] + bottom[0] + data2[N_side-1][0] + data2[N_side-3][0] + data2[N_side-2][1] + data2[N_side-2][2] + data2[N_side-4][0]) / 7.0;
            data2_new[N_side-1][1] = (data2[N_side-1][1] + bottom[1] + data2[N_side-2][1] + bottom[N_side+1] + data2[N_side-1][0] + data2[N_side-3][1] + data2[N_side-1][2] + data2[N_side-1][3]) / 8.0;
            data2_new[N_side-2][1] = (data2[N_side-2][1] + bottom[1] + data2[N_side-1][1] + data2[N_side-3][1] + data2[N_side-2][0] + data2[N_side-2][2] + data2[N_side-4][1] + data2[N_side-2][3]) / 8.0;
        }

        if(has_bottom && !has_right){
            data2_new[N_side-1][N_side-1] = (data2[N_side-1][N_side-1] + bottom[N_side-1] + data2[N_side-2][N_side-1] + bottom[2*N_side-1] + data2[N_side-1][N_side-2] + data2[N_side-3][N_side-1] + data2[N_side-1][N_side-3]) / 7.0;
            data2_new[N_side-2][N_side-1] = (data2[N_side-2][N_side-1] + bottom[N_side-1] + data2[N_side-1][N_side-1] + data2[N_side-3][N_side-1] + data2[N_side-2][N_side-2] + data2[N_side-2][N_side-3] + data2[N_side-4][N_side-1]) / 7.0;
            data2_new[N_side-1][N_side-2] = (data2[N_side-1][N_side-2] + bottom[N_side-2] + data2[N_side-2][N_side-2] + bottom[2*N_side-2] + data2[N_side-1][N_side-1] + data2[N_side-3][N_side-2] + data2[N_side-1][N_side-3] + data2[N_side-1][N_side-4]) / 8.0;
            data2_new[N_side-2][N_side-2] = (data2[N_side-2][N_side-2] + bottom[N_side-2] + data2[N_side-1][N_side-2] + data2[N_side-3][N_side-2] + data2[N_side-2][N_side-1] + data2[N_side-2][N_side-3] + data2[N_side-4][N_side-2] + data2[N_side-2][N_side-4]) / 8.0;
        }

        if(has_right && !has_top){
            data2_new[0][N_side-1] = (data2[0][N_side-1] + data2[1][N_side-1] + data2[0][N_side-2] + data2[0][N_side-3] + data2[2][N_side-1] + right[0] + right[N_side]) / 7.0;
            data2_new[1][N_side-1] = (data2[1][N_side-1] + data2[0][N_side-1] + data2[2][N_side-1] + data2[1][N_side-2] + data2[1][N_side-3] + data2[3][N_side-1] + right[1] + right[N_side+1]) / 8.0;
            data2_new[0][N_side-2] = (data2[0][N_side-2] + data2[1][N_side-2] + data2[0][N_side-1] + data2[0][N_side-3] + data2[2][N_side-2] + right[0] + data2[0][N_side-4]) / 7.0;
            data2_new[1][N_side-2] = (data2[1][N_side-2] + data2[0][N_side-2] + data2[2][N_side-2] + data2[1][N_side-1] + data2[1][N_side-3] + data2[3][N_side-2] + right[1] + data2[1][N_side-4]) / 8.0;
        }

        if(has_right && !has_bottom){
            data2_new[N_side-1][N_side-1] = (data2[N_side-1][N_side-1] + data2[N_side-2][N_side-1] + data2[N_side-1][N_side-2] + data2[N_side-1][N_side-3] + data2[N_side-3][N_side-1] + right[N_side-1] + right[2*N_side-1]) / 7.0;
            data2_new[N_side-2][N_side-1] = (data2[N_side-2][N_side-1] + data2[N_side-1][N_side-1] + data2[N_side-3][N_side-1] + data2[N_side-2][N_side-2] + data2[N_side-2][N_side-3] + data2[N_side-4][N_side-1] + right[N_side-2] + right[2*N_side-2]) / 8.0;
            data2_new[N_side-1][N_side-2] = (data2[N_side-1][N_side-2] + data2[N_side-2][N_side-2] + data2[N_side-1][N_side-1] + data2[N_side-1][N_side-3] + data2[N_side-3][N_side-2] + right[N_side-1] + data2[N_side-1][N_side-4]) / 7.0;
            data2_new[N_side-2][N_side-2] = (data2[N_side-2][N_side-2] + data2[N_side-1][N_side-2] + data2[N_side-3][N_side-2] + data2[N_side-2][N_side-1] + data2[N_side-2][N_side-3] + data2[N_side-4][N_side-2] + right[N_side-2] + data2[N_side-2][N_side-4]) / 8.0;
        }

        if(has_left && !has_top){
            data2_new[0][0] = (data2[0][0] + data2[1][0] + data2[0][1] + data2[0][2] + data2[2][0] + left[0] + left[N_side]) / 7.0;
            data2_new[1][0] = (data2[1][0] + data2[0][0] + data2[2][0] + data2[1][1] + data2[1][2] + data2[3][0] + left[1] + left[N_side+1]) / 8.0;
            data2_new[0][1] = (data2[0][1] + data2[1][1] + data2[0][0] + data2[0][2] + data2[2][1] + data2[0][3] + left[0]) / 7.0;
            data2_new[1][1] = (data2[1][1] + data2[0][1] + data2[2][1] + data2[1][0] + data2[1][2] + data2[3][1] + left[1] + data2[1][3]) / 8.0;
        }

        if(has_left && !has_bottom){
            data2_new[N_side-1][0] = (data2[N_side-1][0] + data2[N_side-2][0] + data2[N_side-1][1] + data2[N_side-1][2] + data2[N_side-3][0] + left[N_side-1] + left[2*N_side-1]) / 7.0;
            data2_new[N_side-2][0] = (data2[N_side-2][0] + data2[N_side-1][0] + data2[N_side-3][0] + data2[N_side-2][1] + data2[N_side-2][2] + data2[N_side-4][0] + left[N_side-2] + left[2*N_side-2]) / 8.0;
            data2_new[N_side-1][1] = (data2[N_side-1][1] + data2[N_side-2][1] + data2[N_side-1][0] + data2[N_side-1][2] + data2[N_side-3][1] + data2[N_side-1][3] + left[N_side-1]) / 7.0;
            data2_new[N_side-2][1] = (data2[N_side-2][1] + data2[N_side-1][1] + data2[N_side-3][1] + data2[N_side-2][0] + data2[N_side-2][2] + data2[N_side-4][1] + left[N_side-2] + data2[N_side-2][3]) / 8.0;
        }

        data2 = data2_new;

        // Reset request and status count for next iteration
        request_count = 0;
        status_count = 0;
    }
    
    eTime = MPI_Wtime();    // End timer

    double time = eTime - sTime;    
    double max_time;
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);    // Get the maximum time taken by any process
    if (rank == 0) {
        printf("Time taken: %f\n", max_time);
    }

    write_matrix_to_csv(data2, N_side, rank);

    for (int i = 0; i < N_side; i++){
        free(data[i]);
        free(data2[i]);
        // free(data2_new[i]);
    }

    free(data);
    free(data2);
    // free(data2_new);
    MPI_Finalize();
    return 0;
}