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
#include <string>
#include <vector>


class SimpleTensor {
    public:
        SimpleTensor(int* shape, int dimension); // dimension for Dim of tensor
        ~SimpleTensor();
        // setters and getters
        void setShape(int* shape, int dimension);
        std::vector<int> getShape();
        float* getBuffer();
        std::vector<int> getStride();


    private:
        int dimension = 0;
        int size = 1; // set with the constructor
        float* dataBuffer; // CUDA memory
        std::vector<int> shape; // CPU
        std::vector<int> stride;
};