#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

const int N = 1000000;
const int DIMENS = 100;
const int ITERATIONS = 100;

double euclideanDistances(double* point, double* center){
    double dist = 0;
    for (int j = 0; j < DIMENS; j++){

        dist += (center[j] - point[j]) * (center[j] - point[j]);

    }
    return sqrt(dist);
}


int main(int argc, char* argv[]){

    double start = clock();
    double* points = (double*)malloc(sizeof(double)*N*DIMENS);
    double * distances = (double*)malloc(sizeof(double)*N);
    double* center = (double*)malloc(sizeof(double)*DIMENS);

    if (points == NULL){
        printf("NULL POINTS\n");
        exit(1);
    }

    for (int j = 0; j < DIMENS; j++){
        center[j] = 500;
    }

    printf("First point center is %f\n", center[0]);

    for (int i = 0; i < N; i++){
        for (int j = 0; j < DIMENS; j++){
            points[i*DIMENS + j] = 100;
        }
    }

    double maxDist = __DBL_MIN__;
    double minDist = __DBL_MAX__;
    int pmx;

    // Calc distances
    for (int z = 0; z < ITERATIONS; z++){
        for (int i = 0; i < N; i++){
        double dst = euclideanDistances(&points[i*DIMENS], center);
        if (dst < minDist){
            minDist = dst;
        }
        if (dst > maxDist){
            maxDist = dst;
            pmx = i;
        }
        }
    }
    

    printf("MIn dist is %f\n", minDist);
    
    printf("Max dist is %f @ %d with %f\n", maxDist, pmx, points[pmx]);
    double end = clock();
    printf("Time: %f\n", (end - start)/CLOCKS_PER_SEC);
    return 0;
}
