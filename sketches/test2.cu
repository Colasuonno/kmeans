#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

// CUDA kernel per calcolare la distanza minima tra ogni punto e i centri (parallelizzato sui centri)
__global__ void compute_min_distances_v2(float *points, float *centers, float *min_distances, int n, int k, int d) {
    extern __shared__ float shared_dist[]; // Memoria condivisa per la riduzione

    int point_idx = blockIdx.x;           // Ogni blocco elabora un punto
    int center_idx = threadIdx.x;         // Ogni thread elabora un centro

    if (point_idx < n) {
        float dist = 0.0f;

        if (center_idx < k) {
            // Calcolo della distanza di un punto rispetto a un centro
            for (int l = 0; l < d; l++) {
                float diff = points[point_idx * d + l] - centers[center_idx * d + l];
                dist += diff * diff;
            }
            dist = sqrtf(dist);
        }

        // Salva la distanza calcolata nella memoria condivisa
        shared_dist[center_idx] = (center_idx < k) ? dist : INFINITY;
        __syncthreads();

        // Riduzione parallela per trovare il minimo
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (center_idx < stride) {
                shared_dist[center_idx] = fminf(shared_dist[center_idx], shared_dist[center_idx + stride]);
            }
            __syncthreads();
        }

        // Il thread 0 del blocco salva il minimo globale
        if (center_idx == 0) {
            min_distances[point_idx] = shared_dist[0];
        }
    }
}

// Funzione per generare numeri casuali tra 0 e 1
void generate_random_points(float *data, int count, int dimensions) {
    for (int i = 0; i < count * dimensions; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    // Parametri
    int n = 1000;   // Numero di punti
    int k = 10;     // Numero di centri
    int d = 3;      // Dimensioni

    size_t points_size = n * d * sizeof(float);
    size_t centers_size = k * d * sizeof(float);
    size_t distances_size = n * sizeof(float);

    // Allocazione della memoria per punti e centri (host)
    float *h_points = (float *)malloc(points_size);
    float *h_centers = (float *)malloc(centers_size);
    float *h_min_distances = (float *)malloc(distances_size);

    // Generazione dei punti e dei centri casuali
    generate_random_points(h_points, n, d);
    generate_random_points(h_centers, k, d);

    // Allocazione della memoria per punti e centri (device)
    float *d_points, *d_centers, *d_min_distances;
    cudaMalloc((void **)&d_points, points_size);
    cudaMalloc((void **)&d_centers, centers_size);
    cudaMalloc((void **)&d_min_distances, distances_size);

    // Copia dei dati dal host al device
    cudaMemcpy(d_points, h_points, points_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centers, h_centers, centers_size, cudaMemcpyHostToDevice);

    // Configurazione dei thread e dei blocchi
    int threads_per_block = 256; // Numero massimo di thread per blocco (pari a k o inferiore)
    int num_blocks = n;          // Ogni blocco elabora un punto

    // Esecuzione del kernel
    compute_min_distances_v2<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(d_points, d_centers, d_min_distances, n, k, d);

    // Copia dei risultati dal device al host
    cudaMemcpy(h_min_distances, d_min_distances, distances_size, cudaMemcpyDeviceToHost);

    // Stampa dei risultati
    printf("Distanze minime:\n");
    for (int i = 0; i < n; i++) {
        printf("Punto %d: %f\n", i, h_min_distances[i]);
    }

    // Pulizia della memoria
    free(h_points);
    free(h_centers);
    free(h_min_distances);
    cudaFree(d_points);
    cudaFree(d_centers);
    cudaFree(d_min_distances);

    return 0;
}
