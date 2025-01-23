#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

// CUDA kernel per calcolare la distanza minima tra ogni punto e i centri
__global__ void compute_min_distances(float *points, float *centers, float *min_distances, int n, int k, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float min_dist = INFINITY;

        for (int j = 0; j < k; j++) {
            float dist = 0.0f;
            for (int l = 0; l < d; l++) {
                float diff = points[idx * d + l] - centers[j * d + l];
                dist += diff * diff;
            }
            dist = sqrtf(dist);

            if (dist < min_dist) {
                min_dist = dist;
            }
        }

        min_distances[idx] = min_dist;
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
    int k = 100;     // Numero di centri
    int d = 100;      // Dimensioni

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
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    // Esecuzione del kernel
    compute_min_distances<<<num_blocks, threads_per_block>>>(d_points, d_centers, d_min_distances, n, k, d);

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