#include <cuda_runtime.h> // cuda mem alloc
#include <stdio.h> // printf
#include <stdlib.h> // malloc C-compat


__global__ void helloKernel(float* a, float *b, float* c) {

    // a, b, c are float arr
    printf("kernel 1\n");

    printf("Block # %d, thread count per block #: %d, thread index within block: %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    
    // lets do repeated tasks
    int i = blockIdx.x * blockDim.x + threadIdx.x; // the exact thread that i have over all my blocks and all my threads
    // say i have 3 blocks * 32 threads = 96 threads, this will identify my exact thread and display what i need on that
    // itll add c[x] = a[x] + b[x], that way, it runs those all in parallel and is very efficient compared to normal O(N) loops.
    c[i] = a[i] + b[i];

}

int main() {

    float arr[96] = {0};

    float arrTwo[96] = {0};

    float arrThree[96];


    helloKernel<<<3, 32>>>(arr, arrTwo, arrThree);

}