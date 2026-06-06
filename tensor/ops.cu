// operations on tensors with GPU

#include "tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>


template <typename T>
__global__ void addKernel(T* a, T* b, T* c, int N) {
    // simple add op

    int i = blockIdx.x * blockDim.x + threadIdx.x; // calculating the exact thread ur located in

    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

template <typename T>
// CPU add
SimpleTensor<T> add(SimpleTensor<T>& a, SimpleTensor<T>& b) {
    
    // valid input shapes
    if (a.getDimension() != b.getDimension()) {
        throw std::invalid_argument("shape sizes arent the same!");
    }

    // check shapes themselves against each other

    for (int i = 0; i < a.getDimension(); i++) {
        if (a.getShape()[i] != b.getShape()[i]) {
            throw std::invalid_argument("shape numbers aren't the same");
        }
    }

    // create output Tensor
    SimpleTensor<T> outputTensor(a.getShape(), a.getDimension()); // this will Have a data buffer here

    // launch kernel, where itll cudaMalloc to the place of the output tensor
    int threads = 256; // default block thread count
    int blocks = (a.getSize() + threads - 1) / threads; // esssentially a ceil function for block counts

    // constructors manage mem!
    addKernel<<<blocks, threads>>>(a.getBuffer(), b.getBuffer(), outputTensor.getBuffer(), a.getSize());


    return outputTensor;
}

