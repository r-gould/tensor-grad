#include "binary.hpp"
#include <cassert>
#include <stdexcept>

template <class T>
Tensor<T>& Binary<T>::forward(Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    Tensor<T>& out{ _forward(tensor1, tensor2) };

    this->cached_args = { &tensor1, &tensor2 };
    out.set_grad_info(this->cached_args, this);
    return out;
}

template <class T>
std::vector<Tensor<T>> Binary<T>::backward(Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    assert((&tensor1 == this->cached_args[0]) && (&tensor2 == this->cached_args[1]));

    if(this->done_backward)
        return this->cached_grads;

    this->cached_grads = { _backward(tensor1, tensor2) };
    this->done_backward = true;
    return this->cached_grads;
}

template <class T>
Tensor<T>& Binary<T>::_forward(Tensor<T>& tensor)
{
    throw std::runtime_error("Binary operation does not support unary arguments.");
}

template <class T>
std::vector<Tensor<T>> Binary<T>::_backward(Tensor<T>& tensor)
{
    throw std::runtime_error("Binary operation does not support unary arguments.");
}

template <class T>
Tensor<T>& Binary<T>::forward(Tensor<T>& tensor)
{
    throw std::runtime_error("Binary operation does not support unary arguments.");
}

template <class T>
std::vector<Tensor<T>> Binary<T>::backward(Tensor<T>& tensor)
{
    throw std::runtime_error("Binary operation does not support unary arguments.");
}

template <class T>
Tensor<T>& Binary<T>::forward(std::vector<Tensor<T>*> args)
{
    assert(args.size() == 2);
    return forward(*(args[0]), *(args[1]));
}

template <class T>
std::vector<Tensor<T>> Binary<T>::backward(std::vector<Tensor<T>*>& args)
{
    assert(args.size() == 2);
    return backward(*(args[0]), *(args[1]));
}

// Template declarations

template class Binary<int>;
template class Binary<double>;
template class Binary<long>;
template class Binary<long long>;