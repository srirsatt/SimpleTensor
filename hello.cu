#include <cuda_runtime.h> // cuda mem alloc
#include <stdio.h> // printf
#include <stdlib.h> // malloc C-compat* 
#include <iostream>

// for rand nums

#include <cstdlib>
#include <ctime>


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

    //printf("blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d", blockIdx.x, blockDim.x, threadIdx.x);
    //printf("c[i] is %f", c[i]);

}

__global__ void parallelMul(int* a, int* b, int* c, int N) {
    // simply a practice in data size restriction. what if your data isnt x32?
    // N - data size

    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread of total threads

    // say N = 1000, for 16 blocks @ 64 threads

    if (i < N) {
        c[i] = a[i] * b[i];
    }


}

int main() {
    //helloKernel<<<3, 32>>>();

    // lets setup GPU arrays and malloc / free / cudaMemCpy for parallelAdd()

    // need a var to declare size of elements
    int N = 96;

    float cpu_a[N];
    float cpu_b[N];
    // first - make CPU memory
    for (int j = 0; j < 96; j++) {
        cpu_a[j] = 1;
        cpu_b[j] = 1;
    }
    float cpu_c[96]; // copy back from d_c to cpu_c later using cudaMemcpy



    float* d_a;
    cudaMalloc(&d_a, N * sizeof(float)); // object ref, size of arr

    float* d_b;
    cudaMalloc(&d_b, N * sizeof(float));

    float* d_c;
    cudaMalloc(&d_c, N*sizeof(float));

    // copy elements via cudaMemCpy

    cudaMemcpy(d_a, cpu_a, N*sizeof(float), cudaMemcpyHostToDevice); // cpu to gpu mem copy
    cudaMemcpy(d_b, cpu_b, N*sizeof(float), cudaMemcpyHostToDevice); // arr b copy

    // run kernel

    parallelAdd<<<3, 32>>>(d_a, d_b, d_c);

    // copy back over

    cudaMemcpy(cpu_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 96; i++) {
        // print of cpu_c back to me
        std::cout << cpu_c[i] << std::endl;
    }




    cudaDeviceSynchronize();

    // parallelAdd(a, b, c); // going thru cudaMalloc & cudaFree through this later
    // simple print

    // free all ur memory

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // second kernel - parallelMul

    int mulOne[1000];
    int mulTwo[1000];

    srand(time(NULL));

    for (int i = 0; i < 1000; i++) {
        mulOne[i] = rand() % 100;
        mulTwo[i] = rand() % 100;
    }


    // now setup the usual
    int mulFinal[1000];

    // malloc CUDA arr, cudaMemcpy

    int* mul_1;
    cudaMalloc(&mul_1, 1000*sizeof(int));
    int* mul_2;
    cudaMalloc(&mul_2, 1000*sizeof(int));
    int* mul_final;
    cudaMalloc(&mul_final, 1000*sizeof(int)); // 4k bytes

    cudaMemcpy(mul_1, mulOne, 1000*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mul_2, mulTwo, 1000*sizeof(int), cudaMemcpyHostToDevice);

    // now run ur computation

    parallelMul<<<16, 64>>>(mul_1, mul_2, mul_final, 1000);

    // copy back over, free mem

    cudaMemcpy(mulFinal, mul_final, 1000*sizeof(int), cudaMemcpyDeviceToHost);

    for (int k = 0; k < 1000; k++) {
        std::cout << mulFinal[k] << std::endl;
    }

    cudaFree(mul_1);
    cudaFree(mul_2);
    cudaFree(mul_final);


}

// compile - nvcc hello.cu -o hello - compiles to hello executable