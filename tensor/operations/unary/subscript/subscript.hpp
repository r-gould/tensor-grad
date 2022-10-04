#ifndef SUBSCRIPT_HPP
#define SUBSCRIPT_HPP

template <class T>
class Unary;

#include "../unary.hpp"
#include "../../../tensor.hpp"
#include <vector>

template <class T>
class Subscript : public Unary<T>
{
protected:
    const std::vector<int> index;
    const int index_size;

    Tensor<T>& _forward(Tensor<T>& tensor) override;
    std::vector<Tensor<T>> _backward(Tensor<T>& tensor) override;

public:
    Subscript(const std::vector<int>& index);
};

#endif