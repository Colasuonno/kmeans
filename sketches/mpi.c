#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <omp.h>


const int N = 1000000;
const int DIMENS = 100;
const int ITERATIONS = 10000;

double euclideanDistances(int rank, double* point, double* center){
    double dist = 0;
    for (int j = 0; j < DIMENS; j++){
        dist += (center[j] - point[j]) * (center[j] - point[j]);
    }
    return sqrt(dist);
}


int main(int argc, char* argv[]){
    int rank,size;
    double start,end;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    int* classMap = NULL;
    int* sendcounts = (int*)malloc(sizeof(int)*size);
    int* displays = (int*)malloc(sizeof(int)*size);

    if (rank == 0){
        classMap = (int*) malloc(sizeof(int)*N);
    }

    int elem_per_proc = N / size;
    int extra = N % size;
    int end_index = rank==size-1 ? elem_per_proc + extra : elem_per_proc;

    int offset = 0;
    // Calculate points per process
    for (int i = 0; i < size; i++){
        sendcounts[i] = (i == size-1 ? elem_per_proc + extra : elem_per_proc);
        displays[i] = offset;
        offset += sendcounts[i];
    }

    int* localClassmap = (int*)malloc(sizeof(int)*sendcounts[rank]);

    for (int j = 0; j < sendcounts[rank]; j++){
        localClassmap[j] = displays[rank] + j;
    }

    MPI_Gatherv(localClassmap, sendcounts[rank], MPI_INT, classMap, sendcounts, displays, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0){
        for (int i = 0; i<  N; i++){
            printf("Classmap %d is %d\n", i, classMap[i]);
        }
        free(classMap);
    }

    free(localClassmap);
    free(sendcounts);
    free(displays);

    MPI_Finalize();


}
