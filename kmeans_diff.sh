#!/bin/bash

# Funzione per mostrare l'uso dello script
function usage() {
    echo "Usage: $0 [Input Filename] [Number of Clusters] [Number of Iterations] [Number of Changes] [Threshold]"
    exit 1
}

# Controllo dei parametri
if [ "$#" -ne 5 ]; then
    usage
fi

# Parametri
INPUT_FILENAME=$1
NUM_CLUSTERS=$2
NUM_ITERATIONS=$3
NUM_CHANGES=$4
THRESHOLD=$5

# File di output
SEQ_OUTPUT="output_sequential.txt"
CUDA_OUTPUT="output_cuda.txt"
OMP_OUTPUT="output_omp.txt"
DIFF_OUTPUT="diff_output.txt"

# Variabile di stato
STATUS=0

# Compilazione dei programmi
echo "Compilazione delle implementazioni..."
make clean && make all || STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo "Errore durante la compilazione."
    exit 1
fi

# Esecuzione della versione sequenziale
echo "Esecuzione della versione sequenziale..."
START_TIME=$(date +%s.%N)
./KMEANS_seq $INPUT_FILENAME $NUM_CLUSTERS $NUM_ITERATIONS $NUM_CHANGES $THRESHOLD $SEQ_OUTPUT || STATUS=$?
END_TIME=$(date +%s.%N)
SEQ_TIME=$(echo "$END_TIME - $START_TIME" | bc)

# Esecuzione della versione CUDA
echo "Esecuzione della versione CUDA..."
START_TIME=$(date +%s.%N)
./KMEANS_cuda $INPUT_FILENAME $NUM_CLUSTERS $NUM_ITERATIONS $NUM_CHANGES $THRESHOLD $CUDA_OUTPUT || STATUS=$?
END_TIME=$(date +%s.%N)
CUDA_TIME=$(echo "$END_TIME - $START_TIME" | bc)

# Esecuzione della versione OMP
echo "Esecuzione della versione OMP..."
START_TIME=$(date +%s.%N)
./KMEANS_cuda $INPUT_FILENAME $NUM_CLUSTERS $NUM_ITERATIONS $NUM_CHANGES $THRESHOLD $OMP_OUTPUT || STATUS=$?
END_TIME=$(date +%s.%N)
OMP_TIME=$(echo "$END_TIME - $START_TIME" | bc)

# Controllo errori di esecuzione
if [ "$STATUS" -ne 0 ]; then
    echo "Errore durante l'esecuzione."
    exit 1
fi

# Confronto degli output
echo "Confronto degli output..."

# Confronta il primo con il secondo
diff $SEQ_OUTPUT $CUDA_OUTPUT > diff1.txt

# Confronta il secondo con il terzo
diff $CUDA_OUTPUT $OMP_OUTPUT > diff2.txt

# Controlla se ci sono differenze
if [ -s diff1.txt ] || [ -s diff2.txt ]; then
    echo "Differenze trovate tra le implementazioni."
    echo "Dettagli in diff1.txt e diff2.txt"
else
    echo "Nessuna differenza trovata tra le implementazioni."
    rm diff1.txt diff2.txt
fi

# Mostra i tempi di esecuzione
echo "Tempo di esecuzione della versione sequenziale: $SEQ_TIME secondi"
echo "Tempo di esecuzione della versione CUDA: $CUDA_TIME secondi"
echo "Tempo di esecuzione della versione OMP: $OMP_TIME secondi"

echo "Script completato."
exit 0
