#include "exp.hpp"
#include "../../../utils/utils.hpp"
#include "../../../tensor.hpp"
#include <cassert>
#include <vector>
#include <cmath>

template <class T>
Exp<T>::Exp(const T base)
    : base{ base }
{}

template <class T>
Tensor<T>& Exp<T>::_forward(Tensor<T>& tensor)
{
    Tensor<T>* out = new Tensor<T>{ tensor.shape, 0 };

    out->modify([&tensor, base=this->base](Tensor<T>& out_tensor, const std::vector<int>& index)
    {
        out_tensor(index) = std::pow(base, tensor(index));
    }
    , tensor.shape);

    return *out;
}

template <class T>
std::vector<Tensor<T>> Exp<T>::_backward(Tensor<T>& tensor)
{
    std::vector<int> grad_shape{ utils::concat_shapes(tensor.shape, tensor.shape) };

    Tensor<T> grad{ grad_shape, 0 };

    grad.modify([&tensor, base=this->base](Tensor<T>& grad_tensor, const std::vector<int> index)
    {
        std::vector<int> total_index{ utils::concat_shapes(index, index) };
        grad_tensor(total_index) = std::log(base) * std::pow(base, tensor(index));
        grad_tensor.non_zero_idxs.push_back(total_index);
    }
    , tensor.shape);

    return { grad };
}


// Template declarations

template class Exp<int>;
template class Exp<double>;
template class Exp<long>;
template class Exp<long long>;