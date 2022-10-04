#ifndef OPERATOR_HPP
#define OPERATOR_HPP

template <class T>
class Tensor;

#include "../tensor.hpp"
#include <vector>

template <class T>
class Operation
{
protected:
    std::vector<Tensor<T>*> cached_args;
    std::vector<Tensor<T>> cached_grads;
    bool done_backward{ false };
    
    // Unary functions
    virtual Tensor<T>& _forward(Tensor<T>& tensor) = 0;
    virtual std::vector<Tensor<T>> _backward(Tensor<T>& tensor) = 0;

    // Binary functions
    virtual Tensor<T>& _forward(Tensor<T>& tensor1, Tensor<T>& tensor2) = 0;
    virtual std::vector<Tensor<T>> _backward(Tensor<T>& tensor1, Tensor<T>& tensor2) = 0;

public:
    // Unary functions
    virtual Tensor<T>& forward(Tensor<T>& tensor) = 0;
    virtual std::vector<Tensor<T>> backward(Tensor<T>& tensor) = 0;

    // Binary functions
    virtual Tensor<T>& forward(Tensor<T>& tensor1, Tensor<T>& tensor2) = 0;
    virtual std::vector<Tensor<T>> backward(Tensor<T>& tensor1, Tensor<T>& tensor2) = 0;
    
    // Generic
    virtual Tensor<T>& forward(std::vector<Tensor<T>*> args) = 0;
    virtual std::vector<Tensor<T>> backward(std::vector<Tensor<T>*>& args) = 0;
};

#endif