#ifndef BROADCAST_HPP
#define BROADCAST_HPP

#include "../unary.hpp"
#include "../../../tensor.hpp"
#include <vector>

template <class T>
class Broadcast : public Unary<T>
{
protected:
    const std::vector<int> new_shape;

    Tensor<T>& _forward(Tensor<T>& scalar) override;
    std::vector<Tensor<T>> _backward(Tensor<T>& scalar) override;

public:
    Broadcast(const std::vector<int>& new_shape);
};

#endif