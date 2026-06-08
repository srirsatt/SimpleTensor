#pragma once
#include <vector>
#include <stdexcept>

enum class ScalarOp {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE
};

enum class ElementWiseOp {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE
};


template <typename T>
__global__ void elementKernel(T* a, T* b, T* c, int N, ElementWiseOp operation);

template <typename T>
SimpleTensor<T> elementOp(SimpleTensor<T>& a, SimpleTensor<T>& b, ElementWiseOp operation);

template <typename T>
__global__ void scalarKernel(T* a, T scalar, T* c, int N, ScalarOp operation);

template <typename T>
SimpleTensor<T> scalarOp(SimpleTensor<T>& a, T scalar, ScalarOp operation);