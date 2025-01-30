#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

int main(int argc, char *argv[]) {
    double start;
   
    int rank, size;
    int N = 100000; // Puoi cambiare N per testare con array più grandi
    double *vector = NULL;
    double *sub_vector;
    double local_sum = 0.0, total_sum = 0.0;
    int chunk_size;

    // Inizializza MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calcola la dimensione del chunk per ogni processo
    chunk_size = N / size;

    // Alloca la memoria per il sotto-vettore che ogni processo calcolerà
    sub_vector = (double *)malloc(chunk_size * sizeof(double));

    // Solo il processo 0 inizializza il vettore
    if (rank == 0) {
        vector = (double *)malloc(N * sizeof(double));
        start = clock();
    }

    // Inizializza i valori locali nei vari processi
    #pragma omp parallel for
    for (int i = rank * chunk_size; i < (rank + 1) * chunk_size; i++) {
        double v_z = 0.0;
        for (int j = 0; j < N; j++) {
            v_z = sqrt(v_z + j);
        }
        sub_vector[i - rank * chunk_size] = v_z;
    }

    // Raccolta dei risultati nel processo 0
    MPI_Gather(sub_vector, chunk_size, MPI_DOUBLE, vector, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Solo il processo 0 stampa il risultato
    if (rank == 0) {
        double total_sum = 0.0;
        for (int i = 0; i < N; i++) {
            total_sum += vector[i];
        }
        double end = clock();
        printf("La somma totale è: %lf\n", total_sum);
        printf("END %f\n", end);
        printf("Ci ho messo %f\n", (double)(end-start)/CLOCKS_PER_SEC);
        free(vector);
    }

    // Libera la memoria allocata
    free(sub_vector);

    // Termina MPI
    MPI_Finalize();


    return 0;
}