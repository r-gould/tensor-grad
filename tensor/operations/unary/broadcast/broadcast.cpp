#include "broadcast.hpp"
#include "../../../tensor.hpp"
#include <vector>
#include <cassert>

template <class T>
Broadcast<T>::Broadcast(const std::vector<int>& new_shape)
    : new_shape{ new_shape }
{}

template <class T>
Tensor<T>& Broadcast<T>::_forward(Tensor<T>& scalar)
{
    T value{ scalar.item() };

    Tensor<T>* out = new Tensor<T>{ this->new_shape, value };
    return *out;
}

template <class T>
std::vector<Tensor<T>> Broadcast<T>::_backward(Tensor<T>& scalar)
{
    std::vector<int> grad_shape = this->new_shape;
    grad_shape.push_back(1);

    Tensor<T> grad{ grad_shape, 1 };
    grad.non_zero_idxs = utils::total_idxs(grad_shape);
    return { grad };
}


// Template declarations

template class Broadcast<int>;
template class Broadcast<double>;
template class Broadcast<long>;
template class Broadcast<long long>;