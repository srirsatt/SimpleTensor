// basic tensor implementation

// tensor should have:

/*

Constructor
Destructor
1. Data Buffer (vector)
2. Shape (3 element arr int)
3. Stride (3 element arr int) - calculated from shape

getters & setters, of course!
*/

#include "tensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>


SimpleTensor::SimpleTensor(std::vector<int> shape, int dimension, float* dataBuffer) {
    // constructor
    // take in the data buffer from cpu mem, copy it to GPU mem, copy shape into the private field and dimension

    dimension_ = dimension;
    shape_ = shape;

    size_ = 1;
    for (int i = 0; i < shape.size(); i++) {
        size_ *= shape[i];
    }

    // compute Stride!
    stride_.resize(dimension_);
    stride_[dimension_ - 1] = 1; // last dim
    for (int i = dimension_ - 2; i >= 0; i--) {
        stride_[i] = stride_[i+1] * shape_[i+1];
    }

    // now i want to copy data buffer through alloc into cuda



    // cudaMalloc
    float* d_buf;
    cudaMalloc(&d_buf, size_*sizeof(float));

    // cudaMemcpy
    cudaMemcpy(d_buf, dataBuffer, size_*sizeof(float), cudaMemcpyHostToDevice); // CPU->GPU memcpy

    dataBuffer_ = d_buf;
}

SimpleTensor::SimpleTensor(std::vector<int> shape, int dimension) {
    // same as before, fill with blanks

    dimension_ = dimension;
    shape_ = shape;

    size_ = 1;
    for (int i = 0; i < shape.size(); i++) {
        size_ *= shape[i];
    }

    // set your stride!
    stride_.resize(dimension_);
    stride_[dimension_ - 1] = 1;

    for (int i = dimension_ - 2; i >= 0; i--) {
        stride_[i] = stride_[i+1] * shape_[i+1];
    }

    float* d_buf;
    cudaMalloc(&d_buf, size_*sizeof(float));

    cudaMemset(d_buf, 0, size_*sizeof(float)); // setting to all 0's, no need to have a Memcpy

    dataBuffer_ = d_buf;
}


SimpleTensor::~SimpleTensor() {
    // destructor

    cudaFree(dataBuffer_); // only thing, everything else takes care
}

void SimpleTensor::setShape(std::vector<int> shape, int dimension) {

}

void SimpleTensor::setBuffer(float* dataBuffer) {

}

std::vector<int> SimpleTensor::getShape() {

}

float* SimpleTensor::getBuffer() {

}

std::vector<int> SimpleTensor::getStride() {

}

