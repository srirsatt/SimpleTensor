// operations on tensors with GPU

#include "tensor.h"
#include "ops.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>


template <typename T>
__global__ void elementKernel(T* a, T* b, T* c, int N, ElementWiseOp operation) {
    // simple add op

    int i = blockIdx.x * blockDim.x + threadIdx.x; // calculating the exact thread ur located in

    if (i < N) {
        switch (operation) {
            case ElementWiseOp::ADD:
                c[i] = a[i] + b[i];
                break;
            case ElementWiseOp::SUBTRACT:
                c[i] = a[i] - b[i];
                break;
            case ElementWiseOp::MULTIPLY:
                c[i] = a[i] * b[i];
                break;
            case ElementWiseOp::DIVIDE:
                c[i] = a[i] / b[i];
                break;
        }
    }
}


/*
template <typename T>
__global__ void subtractKernel(T* a, T* b, T* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        c[i] = a[i] - b[i];
    }
}

template <typename T>
__global__ void multiplyKernel(T* a, T* b, T* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        c[i] = a[i] * b[i];
    }
}

template <typename T>
__global__ void divideKernel(T* a, T* b, T* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        c[i] = a[i] / b[i];
    }
}

*/

template <typename T>
// CPU add
SimpleTensor<T> elementOp(SimpleTensor<T>& a, SimpleTensor<T>& b, ElementWiseOp operation) {
    
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
    elementKernel<<<blocks, threads>>>(a.getBuffer(), b.getBuffer(), outputTensor.getBuffer(), a.getSize(), operation);


    return outputTensor;
}

template <typename T>
__global__ void scalarKernel(T* a, T scalar, T* c, int N, ScalarOp operation) {
    // switch on the operation

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        switch (operation) {
            case ScalarOp::ADD:
                c[i] = a[i] + scalar;
                break;
            case ScalarOp::SUBTRACT:
                c[i] = a[i] - scalar;
                break;
            case ScalarOp::MULTIPLY:
                c[i] = a[i] * scalar;
                break;
            case ScalarOp::DIVIDE:
                c[i] = a[i] / scalar;
                break;
        }
    }
}

template <typename T>
SimpleTensor<T> scalarOp(SimpleTensor<T>& a, T scalar, ScalarOp operation) {
    // outputs operand + Tensor

    SimpleTensor<T> outputTensor(a.getShape(), a.getDimension()); // creates zeroed dataBuf

    int threads = 256;
    int blocks = (a.getSize() + threads - 1) / threads;

    scalarKernel<<<blocks, threads>>>(a.getBuffer(), scalar, outputTensor.getBuffer(), a.getSize(), operation);

    return outputTensor;
}


template SimpleTensor<float> elementOp<float>(SimpleTensor<float>&, SimpleTensor<float>&, ElementWiseOp);
template SimpleTensor<int> elementOp<int>(SimpleTensor<int>&, SimpleTensor<int>&, ElementWiseOp);
template SimpleTensor<float> scalarOp<float>(SimpleTensor<float>&, float, ScalarOp);
template SimpleTensor<int> scalarOp<int>(SimpleTensor<int>&, int, ScalarOp);

