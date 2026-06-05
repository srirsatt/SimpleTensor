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


SimpleTensor::SimpleTensor(int* shape, int dimension) {
    // constructor
}

SimpleTensor::~SimpleTensor() {
    // destructor
}

void SimpleTensor::setShape(int* shape, int dimension) {

}

std::vector<int> SimpleTensor::getShape() {

}

float* SimpleTensor::getBuffer() {

}

std::vector<int> SimpleTensor::getStride() {
    
}

