#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <math.h>
#include <float.h>

int DIMENSION = 100;
int POINTS = 100'000;
int CENTERS = 1000;


// Writing kernel

__global__ void k_euclideanDistance(float* points, float* centers, int* pointClass, float* pointMinDist, int points_numbers, int centers_numbers, int dimensions){

    // L'idea è quella che ogni blocco 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < points_numbers){
    
        // idx is the index of point
        // idy is the index of center

        float minDist = FLT_MAX;
        int currCluster = -1;

        for (int c = 0; c < centers_numbers; c++){
            for (int d = 0; d < dimensions; d++){

                float partialDist = centers[c*dimensions + d] - points[idx * dimensions + d];
                float dist = partialDist * partialDist;

                if (dist < minDist){
                    minDist = dist;
                    currCluster = c;
                }
            }
        }

        pointClass[idx] = currCluster;
        pointMinDist[idx] = minDist;             
        
    } 

}


/*
We have n-dimension points, we need to calculate the distance from a n-dimension center
*/
float euclideanDistance(float* point, float* center, int dimensions){
    float dist = 0.0;

    for (int i = 0; i < dimensions; i++){
        dist += (point[i]-center[i]) * (point[i]-center[i]);
    }

    dist = sqrt(dist);
    return dist;
}

void free_float_float(float** list, int size){
    for (int i = 0; i < size; i++){
        free(list[i]);
    }
    free(list);
}


int main(){

    // Calculate centers
    float* centers = (float*) malloc( sizeof(float) * CENTERS * DIMENSION );
    float* points = (float*) malloc(sizeof(float) * POINTS * DIMENSION);
    
    // Min dist is like
    // mindist[i] = i-min dist from c center
    // pointclass[i] = i-class

    for (int c = 0; c < CENTERS; c++){
        for (int d = 0; d < DIMENSION; d++){
            centers[DIMENSION * c + d] = rand() % 1000;
        } 
    }

    
    for (int p = 0; p < POINTS; p++){
        for (int d = 0; d < DIMENSION; d++){
            points[ p * DIMENSION + d] = rand() % 1000;
        } 
    }

    // CUDA MEM

    // Ogni punto ha n coordinate per le n dimensioni
    // Ogni centro ha n coordinate per le n dimensioni
    // Ogni punto ha una distanza globale (euclidea) rispetto ad ogni centro
    // point -> center

    float* k_points;
    float* k_centers;
    int* k_points_classes;
    float* k_points_distances;

    cudaMalloc((void**)&k_points, sizeof(float)*POINTS*DIMENSION);
    cudaMalloc((void**)&k_centers, sizeof(float)*CENTERS*DIMENSION);
    cudaMalloc((void**)&k_points_classes, sizeof(int)*POINTS);
    cudaMalloc((void**)&k_points_distances, sizeof(float)*POINTS);


    cudaMemcpy(k_points, points, sizeof(float)*POINTS*DIMENSION, cudaMemcpyHostToDevice);
    cudaMemcpy(k_centers, centers, sizeof(float)*CENTERS*DIMENSION, cudaMemcpyHostToDevice);

    // Call kernel
    int threadsPerBlock = 256;

    dim3 block(threadsPerBlock);
    dim3 grid((POINTS + threadsPerBlock - 1) / threadsPerBlock);

    printf("Starting comp for cuda\n");
    double start = clock();

    k_euclideanDistance<<<grid, block>>>(k_points, k_centers, k_points_classes, k_points_distances, POINTS, CENTERS, DIMENSION);
    cudaDeviceSynchronize();

    double end = clock();

    printf("Cuda computation in %f\n", (end - start) / CLOCKS_PER_SEC );

    int* points_classes = (int*)malloc(sizeof(int)*POINTS);
    int* monitor_points_classes = (int*)malloc(sizeof(int)*POINTS);

    cudaMemcpy(points_classes, k_points_classes, sizeof(int)*POINTS, cudaMemcpyDeviceToHost);


    start = clock();

    for (int p = 0; p < POINTS; p++){
        float minDist = FLT_MAX;
        int currCluster = -1;

        for (int c = 0; c < CENTERS; c++){
            for (int d = 0; d < DIMENSION; d++){

                float partialDist = centers[c*DIMENSION + d] - points[p * DIMENSION + d];
                float dist = partialDist * partialDist;

                if (dist < minDist){
                    minDist = dist;
                    currCluster = c;
                }
            }
        }

        monitor_points_classes[p] = currCluster;

    }

    end = clock();

    printf("SEQ computation in %f\n", (end - start) / CLOCKS_PER_SEC );
    printf("Checking Result... :)\n");

    for (int p = 0; p < POINTS; p++){
        if (monitor_points_classes[p] != points_classes[p]){
            printf("THERE IS A PROBLEM IN #%i\n", p);
        }
    }

    cudaFree(k_points);
    cudaFree(k_centers);
    cudaFree(k_points_classes);
    cudaFree(k_points_distances);

    free(points_classes);
    free(points);
    free(centers);
    free(monitor_points_classes);
    

    return 1;

}
