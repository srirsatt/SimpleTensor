#pragma once
#include <vector>
#include <stdexcept>

enum class ScalarOp {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE
};


template <typename T>
__global__ void addKernel(T* a, T* b, T* c, int N);

template <typename T>
SimpleTensor<T> add(SimpleTensor<T>& a, SimpleTensor<T>& b);

template <typename T>
__global__ void scalarKernel(T* a, T scalar, T* c, int N, ScalarOp operation);

template <typename T>
SimpleTensor<T> scalarOp(SimpleTensor<T>& a, T scalar, ScalarOp operation);