#pragma once
#include <vector>
#include <stdexcept>

template <typename T>
__global__ void addKernel(T* a, T* b, T* c, int N);

template <typename T>
SimpleTensor<T> add(SimpleTensor<T>& a, SimpleTensor<T>& b);