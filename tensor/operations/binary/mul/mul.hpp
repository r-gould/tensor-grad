#ifndef MUL_HPP
#define MUL_HPP

template <class T>
class Binary;

#include "../binary.hpp"
#include "../../../tensor.hpp"

template <class T>
class Mul : public Binary<T>
{
protected:
    Tensor<T>& _forward(Tensor<T>& tensor1, Tensor<T>& tensor2) override;
    std::vector<Tensor<T>> _backward(Tensor<T>& tensor1, Tensor<T>& tensor2) override;
};

#endif