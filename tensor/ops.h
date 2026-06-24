#pragma once
#include <stdexcept>
#include <vector>

enum class ScalarOp { ADD, SUBTRACT, MULTIPLY, DIVIDE };

enum class ElementWiseOp { ADD, SUBTRACT, MULTIPLY, DIVIDE };

enum class ReduceOp { SUM, MEAN, MAX, MIN, PRODUCT };

template <typename T>
__global__ void elementKernel(T *a, T *b, T *c, int N, ElementWiseOp operation);

template <typename T>
SimpleTensor<T> elementOp(SimpleTensor<T> &a, SimpleTensor<T> &b,
                          ElementWiseOp operation);

template <typename T>
__global__ void scalarKernel(T *a, T scalar, T *c, int N, ScalarOp operation);

template <typename T>
SimpleTensor<T> scalarOp(SimpleTensor<T> &a, T scalar, ScalarOp operation);

template <typename T>
__global__ void reduceAdd(T* input, T* output, int N, ReduceOp operation);

template <typename T>
SimpleTensor<T> reduceOp(SimpleTensor<T>& a, ReduceOp operation);

template <typename T>
__global__ void naiveMatmul(T* a, T* b, T* output, int N, int M, int K);

template <typename T>
SimpleTensor<T> matmul(SimpleTensor<T> &a, SimpleTensor<T> &b);