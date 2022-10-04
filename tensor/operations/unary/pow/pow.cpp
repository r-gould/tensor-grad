#include "pow.hpp"
#include "../../../utils/utils.hpp"
#include "../../../tensor.hpp"
#include <cassert>
#include <vector>
#include <cmath>

template <class T>
Pow<T>::Pow(const T power)
    : power{ power }
{}

template <class T>
Tensor<T>& Pow<T>::_forward(Tensor<T>& tensor)
{
    Tensor<T>* out = new Tensor<T>{ tensor.shape, 0 };

    out->modify([&tensor, power=this->power](Tensor<T>& out_tensor, const std::vector<int>& index)
    {
        out_tensor(index) = std::pow(tensor(index), power);
    }
    , tensor.shape);

    return *out;
}

template <class T>
std::vector<Tensor<T>> Pow<T>::_backward(Tensor<T>& tensor)
{
    std::vector<int> grad_shape{ utils::concat_shapes(tensor.shape, tensor.shape) };

    Tensor<T> grad{ grad_shape, 0 };

    grad.modify([&tensor, power=this->power](Tensor<T>& grad_tensor, const std::vector<int> index)
    {
        std::vector<int> total_index{ utils::concat_shapes(index, index) };
        grad_tensor(total_index) = power * std::pow(tensor(index), power-1);
        grad_tensor.non_zero_idxs.push_back(total_index);
    }
    , tensor.shape);

    return { grad };
}


// Template declarations

template class Pow<int>;
template class Pow<double>;
template class Pow<long>;
template class Pow<long long>;