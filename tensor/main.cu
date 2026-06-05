#include "tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// test file for tensor.cu

int main() {
    float data[] = {1, 2, 3, 4, 5, 6}; // 6 element vector, 
    SimpleTensor tensorOne({2, 3}, 2, data);

    // CPU memcpy for testing

    float out[6];
    cudaMemcpy(out, tensorOne.getBuffer(), 6*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 6; i++) {
        std::cout << out[i] << std::endl; // verify the data copied
    }

    for (int s : tensorOne.getStride()) {
        std::cout << s << std::endl; // 3, 1
    }

    for (int s : tensorOne.getShape()) {
        std::cout << s << std::endl; // 2, 3
    }

    return 0;
}