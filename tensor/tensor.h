// tensor.h

/*
Class Declaration

Constructor
Destructor
1. Data Buffer (vector)
2. Shape (3 element arr int)
3. Stride (3 element arr int) - calculated from shape

getters & setters, of course!
*/

#pragma once // replaces ifndef and define, endif
#include <vector>
#include <stdexcept>

template <typename T>
class SimpleTensor {
    public:
        SimpleTensor(std::vector<int> shape, int dimension, T* dataBuffer);
        SimpleTensor(std::vector<int> shape, int dimension); // dimension for Dim of tensor
        ~SimpleTensor();
        // setters and getters
        void reshape(std::vector<int> shape, int dimension);
        void setBuffer(T* dataBuffer, int size); // copy from cpu to gpu mem
        void print();
        std::vector<int> getShape();
        T* getBuffer();
        std::vector<int> getStride();
        std::vector<T> toHost();
        


    private:
        int dimension_;
        int size_; // set with the constructor
        T* dataBuffer_; // CUDA memory
        std::vector<int> shape_; // CPU
        std::vector<int> stride_;
};