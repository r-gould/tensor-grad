#ifndef MATMUL_HPP
#define MATMUL_HPP

#include "../binary.hpp"
#include "../../../tensor.hpp"

template <class T>
class MatMul : public Binary<T>
{
protected:
    Tensor<T>& _forward(Tensor<T>& tensor1, Tensor<T>& tensor2) override;
    std::vector<Tensor<T>> _backward(Tensor<T>& tensor1, Tensor<T>& tensor2) override;
};

#endif