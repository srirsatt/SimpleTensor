#include "tensor.h"
#include "ops.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <chrono>

// test file for tensor.cu

void benchmarkMatmul() {
    int M = 1024, N = 1024, K = 1024;
    int size = M * K;

    float* hostA = new float[M * K];
    float* hostB = new float[K * N];
    for (int i = 0; i < M * K; i++) hostA[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) hostB[i] = (float)rand() / RAND_MAX;

    SimpleTensor<float> A({M, K}, 2, hostA);
    SimpleTensor<float> B({K, N}, 2, hostB);

    // warmup
    SimpleTensor<float> warmup = tiledMatmul(A, B);
    cudaDeviceSynchronize();

    // naive
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        SimpleTensor<float> C = naiveMatmul(A, B);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double naiveMs = std::chrono::duration<double, std::milli>(end - start).count() / 10;

    // tiled
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        SimpleTensor<float> C = tiledMatmul(A, B);
        cudaDeviceSynchronize();
    }
    end = std::chrono::high_resolution_clock::now();
    double tiledMs = std::chrono::duration<double, std::milli>(end - start).count() / 10;

    printf("=== Matmul Benchmark (1024x1024) ===\n");
    printf("Naive:  %.2f ms\n", naiveMs);
    printf("Tiled:  %.2f ms\n", tiledMs);
    printf("Speedup: %.2fx\n", naiveMs / tiledMs);

    delete[] hostA;
    delete[] hostB;
}

void benchmarkAutograd() {
    int M = 512, N = 512, K = 512;
    float* hostA = new float[M * K];
    float* hostB = new float[K * N];
    for (int i = 0; i < M * K; i++) hostA[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) hostB[i] = (float)rand() / RAND_MAX;

    SimpleTensor<float> A({M, K}, 2, hostA, true);
    SimpleTensor<float> B({K, N}, 2, hostB, true);

    auto start = std::chrono::high_resolution_clock::now();
    SimpleTensor<float> C = tiledMatmul(A, B);
    SimpleTensor<float> loss = reduceOp(C, ReduceOp::SUM);
    loss.backward();
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    printf("=== Autograd Benchmark (512x512 matmul + backward) ===\n");
    printf("Forward + Backward: %.2f ms\n", ms);

    delete[] hostA;
    delete[] hostB;
}

void benchmarkCPUvsGPU() {
    int M = 1024, N = 1024, K = 1024;

    float* hostA = new float[M * K];
    float* hostB = new float[K * N];
    float* hostC = new float[M * N];

    for (int i = 0; i < M * K; i++) hostA[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) hostB[i] = (float)rand() / RAND_MAX;

    // CPU matmul
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += hostA[i * K + k] * hostB[k * N + j];
            }
            hostC[i * N + j] = sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(end - start).count();

    // GPU tiled
    SimpleTensor<float> A({M, K}, 2, hostA);
    SimpleTensor<float> B({K, N}, 2, hostB);

    // warmup
    SimpleTensor<float> warmup = tiledMatmul(A, B);
    cudaDeviceSynchronize();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        SimpleTensor<float> C = tiledMatmul(A, B);
        cudaDeviceSynchronize();
    }
    end = std::chrono::high_resolution_clock::now();
    double gpuMs = std::chrono::duration<double, std::milli>(end - start).count() / 10;

    printf("=== CPU vs GPU Benchmark (1024x1024) ===\n");
    printf("CPU:     %.2f ms\n", cpuMs);
    printf("GPU:     %.2f ms\n", gpuMs);
    printf("Speedup: %.2fx\n", cpuMs / gpuMs);

    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
}

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

    benchmarkMatmul();
    benchmarkAutograd();
    benchmarkCPUvsGPU();

    return 0;
}
