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
        SimpleTensor(std::vector<int> shape, int dimension, T* dataBuffer, bool requiresGrad = false);
        SimpleTensor(std::vector<int> shape, int dimension, bool requiresGrad = false); // dimension for Dim of tensor
        ~SimpleTensor();
        // setters and getters
        void reshape(std::vector<int> shape, int dimension);
        void setBuffer(T* dataBuffer, int size); // copy from cpu to gpu mem
        void setRequiresGrad(bool input);
        void print();
        std::vector<int> getShape();
        T* getBuffer();
        T* getGradBuffer();
        std::vector<int> getStride();
        std::vector<T> toHost();
        int getDimension();
        int getSize();
        bool getRequiresGrad();
        void backward(); // backward pass for autograd
        


    private:
        int dimension_;
        int size_; // set with the constructor
        bool requiresGrad_; // to check if a tensor has AutoGrad computations linked
        T* dataBuffer_; // CUDA memory
        T* gradBuffer_; // holds gradients from AutoGrad production
        std::vector<int> shape_; // CPU
        std::vector<int> stride_;
        std::function<void()> backward_; // backward pass recursive helper - calculates actual gradient values and recursess
};