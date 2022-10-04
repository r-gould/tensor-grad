#ifndef SUM_HPP
#define SUM_HPP

template <class T>
class Unary;

#include "../unary.hpp"
#include "../../../tensor.hpp"
#include <vector>

template <class T>
class Sum : public Unary<T>
{
protected:
    Tensor<T>& _forward(Tensor<T>& tensor) override;
    std::vector<Tensor<T>> _backward(Tensor<T>& tensor) override;
};

#endif