#include "tensor.h"
#include "ops.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// test file for tensor.cu

int main() {
    float data[] = {1, 2, 3, 4, 5, 6}; // 6 element vector, 
    SimpleTensor<float> tensorOne({2, 3}, 2, data);

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

    // grab to host, print resulting array

    for (float num : tensorOne.toHost()) {
        std::cout << num << std::endl;
    }

    // CUDA ops testing
    std::cout << std::endl;

    float data_1[] = {1, 2, 3, 4, 5, 6};
    float data_3[] = {2, 4, 6, 8, 10, 12};

    float scalar = 4;

    SimpleTensor<float> a({2, 3, 1}, 3, data_1);
    SimpleTensor<float> b({2, 3, 1}, 3, data_3);

    //SimpleTensor<float> c = add(a, b);

    SimpleTensor<float> d = scalarOp(a, scalar, ScalarOp::MULTIPLY);

    SimpleTensor<float> e = elementOp(a, b, ElementWiseOp::MULTIPLY);

    SimpleTensor<float> crazy = reduceOp(a, ReduceOp::MAX);

    SimpleTensor<float> testy = reduceOp(a, ReduceOp::MEAN); // 3.5

    //c.print();

    e.print();
    crazy.print();
    testy.print();
    

    return 0;
}
