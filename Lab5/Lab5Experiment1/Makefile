NVCC =  /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all: a.out

a.out: distanceMain.o DistKernel.o
	echo "making..."
	$(NVCC) $^ -o $@

distanceMain.o: distanceMain.cpp DistKernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

DistKernel.o: DistKernel.cu DistKernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

run: a.out
	echo "running..."
	./a.out

clean:
	echo "cleaning..."
	rm -rf *.o a.out
