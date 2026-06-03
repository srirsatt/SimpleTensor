#include <cuda_runtime.h> // cuda mem alloc
#include <stdio.h> // printf
#include <stdlib.h> // malloc C-compatoat* 


__global__ void helloKernel() {

    // a, b, c are float arr
    //printf("kernel 1\n");

    printf("Block # %d, thread count per block #: %d, thread index within block: %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    
    // lets do repeated tasks
    /*
    int i = blockIdx.x * blockDim.x + threadIdx.x; // the exact thread that i have over all my blocks and all my threads
    // say i have 3 blocks * 32 threads = 96 threads, this will identify my exact thread and display what i need on that
    // itll add c[x] = a[x] + b[x], that way, it runs those all in parallel and is very efficient compared to normal O(N) loops.
    c[i] = a[i] + b[i];

    */

}


__global__ void parallelAdd(float* a, float* b, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // calculating, of all threads (blocks * threadCount) - what thread am i at

    // a, b, c - float arrs accessed through C++ pointer logic


    c[i] = a[i] + b[i];

    printf("blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d", blockIdx.x, blockDim.x, threadIdx.x);

}

int main() {
    helloKernel<<<3, 32>>>();

    cudaDeviceSynchronize();

    // parallelAdd(a, b, c); // going thru cudaMalloc & cudaFree through this later
    // simple print
}

// compile - nvcc hello.cu -o hello - compiles to hello executable