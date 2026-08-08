NVCC = nvcc 
FLAGS = -std=c++17 -O2

all: main

main: tensor/main.cu tensor/ops.cu tensor/tensor.cu
	$(NVCC) $(FLAGS) tensor/main.cu tensor/ops.cu tensor/tensor.cu -o main

clean:
	rm -f main