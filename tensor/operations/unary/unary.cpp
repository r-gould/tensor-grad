#include "unary.hpp"
#include <cassert>
#include <stdexcept>

template <class T>
Tensor<T>& Unary<T>::forward(Tensor<T>& tensor)
{
    Tensor<T>& out{ _forward(tensor) };

    this->cached_args = { &tensor };
    out.set_grad_info(this->cached_args, this);
    return out;
}

template <class T>
std::vector<Tensor<T>> Unary<T>::backward(Tensor<T>& tensor)
{
    assert(&tensor == this->cached_args[0]);

    if(this->done_backward)
        return this->cached_grads;

    this->cached_grads = { _backward(tensor) };
    this->done_backward = true;
    return this->cached_grads;
}

template <class T>
Tensor<T>& Unary<T>::_forward(Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    throw std::runtime_error("Unary operation does not support binary arguments.");
}

template <class T>
std::vector<Tensor<T>> Unary<T>::_backward(Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    throw std::runtime_error("Unary operation does not support binary arguments.");
}

template <class T>
Tensor<T>& Unary<T>::forward(Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    throw std::runtime_error("Unary operation does not support binary arguments.");
}

template <class T>
std::vector<Tensor<T>> Unary<T>::backward(Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    throw std::runtime_error("Unary operation does not support binary arguments.");
}

template <class T>
Tensor<T>& Unary<T>::forward(std::vector<Tensor<T>*> args)
{
    assert(args.size() == 1);
    return forward(*(args[0]));
}

template <class T>
std::vector<Tensor<T>> Unary<T>::backward(std::vector<Tensor<T>*>& args)
{
    assert(args.size() == 1);
    return backward(*(args[0]));
}

template class Unary<int>;
template class Unary<double>;
template class Unary<long>;
template class Unary<long long>;