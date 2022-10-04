#include "log.hpp"
#include "../../../utils/utils.hpp"
#include "../../../tensor.hpp"
#include <cassert>
#include <vector>
#include <cmath>

template <class T>
Log<T>::Log(const T base)
    : base{ base }
{}

template <class T>
Tensor<T>& Log<T>::_forward(Tensor<T>& tensor)
{
    Tensor<T>* out = new Tensor<T>{ tensor.shape, 0 };

    out->modify([&tensor, base=this->base](Tensor<T>& out_tensor, const std::vector<int>& index)
    {
        out_tensor(index) = std::log(tensor(index)) / std::log(base);
    }
    , tensor.shape);

    return *out;
}

template <class T>
std::vector<Tensor<T>> Log<T>::_backward(Tensor<T>& tensor)
{
    std::vector<int> grad_shape{ utils::concat_shapes(tensor.shape, tensor.shape) };

    Tensor<T> grad{ grad_shape, 0 };

    grad.modify([&tensor, base=this->base](Tensor<T>& grad_tensor, const std::vector<int> index)
    {
        std::vector<int> total_index{ utils::concat_shapes(index, index) };
        grad_tensor(total_index) = 1.0 / (tensor(index) * std::log(base));
        grad_tensor.non_zero_idxs.push_back(total_index);
    }
    , tensor.shape);

    return { grad };
}


// Template declarations

template class Log<int>;
template class Log<double>;
template class Log<long>;
template class Log<long long>;