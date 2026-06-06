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


class SimpleTensor {
    public:
        SimpleTensor(std::vector<int> shape, int dimension, float* dataBuffer);
        SimpleTensor(std::vector<int> shape, int dimension); // dimension for Dim of tensor
        ~SimpleTensor();
        // setters and getters
        void reshape(std::vector<int> shape, int dimension);
        void setBuffer(float* dataBuffer, int size); // copy from cpu to gpu mem
        void print();
        void reshape(std::vector<int> newShape, int dimension);
        std::vector<int> getShape();
        float* getBuffer();
        std::vector<int> getStride();
        std::vector<float> toHost();
        


    private:
        int dimension_;
        int size_; // set with the constructor
        float* dataBuffer_; // CUDA memory
        std::vector<int> shape_; // CPU
        std::vector<int> stride_;
};