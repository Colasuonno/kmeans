BINS:= mpi greetings omp hybrid

all: $(BINS)

mpi: mpi.c
	mpicc -lm -fopenmp -o mpi mpi.c

greetings: mpi-greetings.c
	mpicc -o greetings mpi-greetings.c

omp: omp-hello.c
	gcc -fopenmp -o omp omp-hello.c

seq: seq.c
	gcc -lm seq.c -o sec

hybrid: hybrid.c
	mpicc -openmp -lgomp -lm -o hybrid hybrid.c

test: $(BINS)
	@echo "Test MPI..."
	@mpirun -np 4 ./mpi
	@mpirun -np 4 ./greetings
	@echo "Test OpenMP..."
	@./omp
	@echo "Test Hybrid..."
	@mpirun -np 4 ./hybrid

clean:
	@rm -f $(BINS)