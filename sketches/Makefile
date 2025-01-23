CC          = /opt/cuda/bin/nvcc
LIBS        = -lm
CFLAGS      = -arch=compute_75 --compiler-bindir=/usr/bin/gcc-13 -DBLUR_SIZE=1 -DBLOCK_SIZE=16

RM=rm -f

EXES=cuda_test test test2

all: $(EXES)

%: %.cu 
	$(CC) $(CFLAGS) -o $@ $@.cu $(LIBS)

clean:
	$(RM) $(EXES)