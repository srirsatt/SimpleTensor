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
#include <stdexcept> // for exceptions


SimpleTensor::SimpleTensor(std::vector<int> shape, int dimension, float* dataBuffer) {
    // constructor
    // take in the data buffer from cpu mem, copy it to GPU mem, copy shape into the private field and dimension

    if (shape.size() != dimension) {
        throw std::invalid_argument("shape size must match dimension count!");
    }

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

    if (shape.size() != dimension) {
        throw std::invalid_argument("shape size must match dimension count!");
    }

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

void SimpleTensor::reshape(std::vector<int> shape, int dimension) {

    // check validity within shape

    if (shape.size() != dimension) {
        throw std::invalid_argument("shape size has to match dimension count!");
    }

    shape_ = shape;
    dimension_ = dimension;

    // calculate new size and verify it matches up, otherwise through an invalid_arg

    int newSize = 1;
    for (int i = 0; i < shape.size(); i++) {
        newSize *= shape[i];
    }

    if (newSize != size_) {
        throw std::invalid_argument("new shape has to properly conform to existing size standards.");
    }

    // recalc stride
    stride_.resize(dimension_);
    stride_[dimension_ - 1] = 1;
    for (int i = dimension_ - 2; i >= 0; i--) {
        stride_[i] = stride_[i+1] * shape_[i+1];
    }
}

void SimpleTensor::setBuffer(float* dataBuffer, int size) {

    // buffer - change in data
    // size check for consistency

    if (size != size_) {
        throw std::invalid_argument("new data buffer size has to match to the existing data buffer size.");
    }

    // new gpu malloc

    // free existing buffer

    cudaFree(dataBuffer_);


    float* d_buf;
    cudaMalloc(&d_buf, size_*sizeof(float));

    cudaMemcpy(d_buf, dataBuffer, size_*sizeof(float), cudaMemcpyHostToDevice);

    dataBuffer_ = d_buf;
}

void SimpleTensor::print() {
    // print method - want to print in the Kernel shape

    // print per matrix, for each "layer" in the matrix

    std::vector<float> data = toHost();

    // now lets print it

    if (dimension_ == 1) {
        std::cout << "1D tensor. printing elements in the one row!" << std::endl;
        for (int i = 0; i < size_; i++) {
            std::cout << data[i] << std::endl;
        }
    } else if (dimension_ == 2) {
        std::cout << "2D Tensor. printing out matrix!" << std::endl;

        for (int i = 0; i < shape_[0]; i++) {
            for (int j = 0; j < shape_[1]; j++) {
                std::cout << data[i * stride_[0] + j * stride_[1]] << " ";
            }
            std::cout << std::endl;
        }
    } else if (dimension_ == 3) {
        std::cout << "3d tensor. printing out matrices!" << std::endl;
        for (int i = 0; i < shape_[0]; i++) {
            for (int j = 0; j < shape_[1]; j++) {
                for (int k = 0; k < shape_[2]; k++) {
                    std::cout << data[i * stride_[0] + j * stride_[1] + k * stride_[2]] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "---" << std::endl;  // separator between matrices
        } 
    } else {
        // all elements simple print
        for (int i = 0; i < data.size(); i++) {
            std::cout << "element " << i << ": " << data[i];
        }
    }
}

std::vector<int> SimpleTensor::getShape() {

    return shape_;

}

float* SimpleTensor::getBuffer() {

    return dataBuffer_; // keep in mind, this returns the Addr to the float arr

}

std::vector<int> SimpleTensor::getStride() {

    return stride_;
}

std::vector<float> SimpleTensor::toHost() {

    std::vector<float> dataBuffer;

    dataBuffer.resize(size_);

    cudaMemcpy(dataBuffer.data(), dataBuffer_, size_*sizeof(float), cudaMemcpyDeviceToHost);


    return dataBuffer;
}

