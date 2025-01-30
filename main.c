#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>


#define ARRAY_SIZE 9999999


int main(int argc, char *argv[]) {
    
    int rank, num_procs;
    unsigned int *full_array = NULL;
    unsigned int *local_array = NULL;
    int local_size;
    int global_min = INT_MAX;
    int local_min = INT_MAX;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0) {

        full_array = malloc(sizeof(unsigned int) * ARRAY_SIZE);
        
        srand(time(NULL) ^ (getpid() << 16));
        
        for (int i = 0; i < ARRAY_SIZE; i++) {
            full_array[i] = rand();
        }

        printf("First 10 array elements:\n");
        for (int i = 0; i < 10; i++) {
            printf("%u\n", full_array[i]);
            }
    }

    local_size = ARRAY_SIZE / num_procs;
    local_array = malloc(sizeof(unsigned int) * local_size);

    MPI_Scatter(full_array, local_size, MPI_UNSIGNED, 
                local_array, local_size, MPI_UNSIGNED, 
                0, MPI_COMM_WORLD);

    printf("Rank %d local array first 5 elements: ", rank);
    for (int i = 0; i < 5; i++) {
        printf("%u ", local_array[i]);
    }
    printf("\n");

    #pragma omp parallel
    {
        #pragma omp for reduction(min:local_min)
        for (int i = 0; i < local_size; i++) {
            if (local_array[i] < local_min) {
                local_min = local_array[i];
            }
        }
    }

    printf("Rank %d local minimum: %d\n", rank, local_min);

    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global Minimum: %d\n", global_min);
    }

    free(local_array);
    if (rank == 0) {
        free(full_array);
    }

    MPI_Finalize();

    return 0;
}