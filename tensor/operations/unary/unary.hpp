#ifndef UNARY_HPP
#define UNARY_HPP

#include "../operation.hpp"
#include "../../tensor.hpp"
#include <vector>

template <class T>
class Unary : public Operation<T>
{
protected:
    virtual Tensor<T>& _forward(Tensor<T>& tensor) = 0;
    virtual std::vector<Tensor<T>> _backward(Tensor<T>& tensor) = 0;

    Tensor<T>& _forward(Tensor<T>& tensor1, Tensor<T>& tensor2) override;
    std::vector<Tensor<T>> _backward(Tensor<T>& tensor1, Tensor<T>& tensor2) override;

public:
    Tensor<T>& forward(Tensor<T>& tensor) override;
    std::vector<Tensor<T>> backward(Tensor<T>& tensor) override;

    Tensor<T>& forward(Tensor<T>& tensor1, Tensor<T>& tensor2) override;
    std::vector<Tensor<T>> backward(Tensor<T>& tensor1, Tensor<T>& tensor2) override;

    Tensor<T>& forward(std::vector<Tensor<T>*> args) override;
    std::vector<Tensor<T>> backward(std::vector<Tensor<T>*>& args) override;
};

#endif