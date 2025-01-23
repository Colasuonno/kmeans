# Analizziamo le performance del SEQ/CUDA/MPI/Pthreads



### (Performance mio pc)
- i5 7600k
- NVIDIA 2070 SUPER



## Analizi sequenziale:

### Test base 

```bash
./KMEANS input_2d.txt 5 100 10 0.001 output_2d.txt
```

Computation: 0.003182 seconds

Termination condition:
Minimum number of changes reached: 335 [500]



```bash
./KMEANS input_10d.txt 5 100 10 0.001 output_2d.txt
```

Computation: 0.000154 seconds

Termination condition:
Minimum number of changes reached: 68 [100]


```bash
./KMEANS input_20d.txt 5 100 10 0.001 output_2d.txt
```

Computation: 0.003667 seconds

Termination condition:
Minimum number of changes reached: 700 [1000]


```bash
./KMEANS input_100d.txt 5 100 10 0.001 output_2d.txt
```

Computation: 0.011023 seconds

Termination condition:
Minimum number of changes reached: 886 [1000]


### Test scalabilità numero di cluster

#### 10 cluster

Computation: 0.027007 seconds

Termination condition:
Minimum number of changes reached: 789 [1000]

#### 100 cluster


Computation: 0.234917 seconds

Termination condition:
Minimum number of changes reached: 755 [1000]


#### 1000 cluster

Computation: 0.027007 seconds

Termination condition:
Minimum number of changes reached: 789 [1000]

#### 10_000 cluster

Computation: 11.575510 seconds

Termination condition:
Minimum number of changes reached: 1 [1000]

#### 100_000 cluster

Computation: 60.416593 seconds

Termination condition:
Centroid update precision reached: 1.17549e-38 [0.001]


## Test scalabilità iterazioni

#### 10 iterazioni

Computation: 60.416593 seconds

Termination condition:
Centroid update precision reached: 1.17549e-38 [0.001]

#### 100 iterazioni

Computation: 0.027124 seconds

Termination condition:
Minimum number of changes reached: 789 [1000]


#### 1000 iterazioni

Computation: 0.027050 seconds

Termination condition:
Minimum number of changes reached: 789 [1000]


#### 10_000 iterazioni

Computation: 0.027145 seconds

Termination condition:
Minimum number of changes reached: 789 [1000]


.... Anche all'infinito le iterazioni vengono cappate dal minimo numero di changes....


### Test sulla threshold

#### Soglia 0.1

Computation: 0.027099 seconds

Termination condition:
Minimum number of changes reached: 789 [1000]


#### Soglia 0.01

Computation: 0.027239 seconds

Termination condition:
Minimum number of changes reached: 789 [1000]


#### Soglia 0.0000.....1


Cappato dalle min changes


### Test scalabilità min changes


#### 1% change

Computation: 0.149648 seconds

Termination condition:
Minimum number of changes reached: 90 [100]


## Test difficile

```bash
./KMEANS_seq test_files/input100D.inp 100000 1000 1 0.001 out.text
```
Computation: 61.196820 seconds

Termination condition:
Centroid update precision reached: 1.17549e-38 [0.001]


