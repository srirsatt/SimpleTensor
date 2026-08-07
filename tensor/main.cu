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

    //float scalar = 4;

    SimpleTensor<float> a({2, 3, 1}, 3, data_1);
    SimpleTensor<float> b({2, 3, 1}, 3, data_3);
    SimpleTensor<float> c({3, 1}, 2, data_1);
    SimpleTensor<float> d({1, 5}, 2, data_3);

    //SimpleTensor<float> c = add(a, b);

    //SimpleTensor<float> d = scalarOp(a, scalar, ScalarOp::MULTIPLY);

    //SimpleTensor<float> e = elementOp(a, b, ElementWiseOp::MULTIPLY);

    SimpleTensor<float> crazy = reduceOp(a, ReduceOp::MAX);

    SimpleTensor<float> testy = reduceOp(a, ReduceOp::MEAN); // 3.5

    SimpleTensor<float> rusheel = tiledMatmul(c, d);

    //c.print();

    /*
    e.print();
    crazy.print();
    testy.print();

    */
    rusheel.print();


    float dataNew[] = {2.0, 3.0};
    float dataNewTwo[] = {4.0, 5.0};

    SimpleTensor<float> A({2}, 1, dataNew, true);
    SimpleTensor<float> B({2}, 1, dataNewTwo, true);

    SimpleTensor<float> C = elementOp(A, B, ElementWiseOp::MULTIPLY);
    C.backward();

    auto gradA = A.toHostGrad();

    for (int i = 0; i < 2; i++) {
        printf("grad_A[%d] = %.2f (expected %.2f)\n", i, gradA[i], dataNew[i]);
    }

    auto gradB = B.toHostGrad();

    for (int i = 0; i < 2; i++) {
        printf("grad_B[%d] = %.2f (expected %.2f)\n", i, gradB[i], dataNew[i]);
    }

    
    float dataF[] = {1.0, 2.0, 3.0, 4.0};
    float dataG[] = {5.0, 6.0, 7.0, 8.0};

    SimpleTensor<float> New({2, 2}, 2, dataF, true);
    SimpleTensor<float> NewTwo({2, 2}, 2, dataG, true);

    SimpleTensor<float> NewC = tiledMatmul(New, NewTwo);

    NewC.backward();

    auto gradNew = New.toHostGrad();
    auto gradNewTwo = NewTwo.toHostGrad();


    printf("grad_New (expected 11, 15, 11, 15):\n");
    for (int i = 0; i < 4; i++) printf("%.2f ", gradNew[i]);
    printf("\n");

    printf("grad_NewTwo (expected 4, 4, 6, 6):\n");
    for (int i = 0; i < 4; i++) printf("%.2f ", gradNewTwo[i]);
    printf("\n");

    float data1337[] = {1.0, 2.0, 3.0, 4.0};
    float data4829[] = {5.0, 6.0, 7.0, 8.0};

    SimpleTensor<float> t7291({2,2}, 2, data1337, true);
    SimpleTensor<float> t5836({2,2}, 2, data4829, true);

    SimpleTensor<float> t9472 = tiledMatmul(t7291, t5836);
    SimpleTensor<float> t2651 = reduceOp(t9472, ReduceOp::SUM);

    t2651.backward();

    auto g3847 = t7291.toHostGrad();
    auto g6194 = t5836.toHostGrad();

    printf("grad_A:\n");
    for (int i = 0; i < 4; i++) printf("%.2f ", g3847[i]);
    printf("\ngrad_B:\n");
    for (int i = 0; i < 4; i++) printf("%.2f ", g6194[i]);

    return 0;
}
