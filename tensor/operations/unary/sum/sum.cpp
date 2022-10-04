#include "sum.hpp"
#include "../../../utils/utils.hpp"
#include "../../../tensor.hpp"
#include <cassert>
#include <vector>
#include <numeric>

template <class T>
Tensor<T>& Sum<T>::_forward(Tensor<T>& tensor)
{
    const std::vector<T>& data{ tensor.data };
    T sum{ std::accumulate(data.begin(), data.end(), static_cast<T>(0)) };
    
    Tensor<T>* out = new Tensor<T>{ std::vector<T>{ sum } };
    return *out;
}

template <class T>
std::vector<Tensor<T>> Sum<T>::_backward(Tensor<T>& tensor)
{
    std::vector<int> out_shape{ tensor.shape };
    out_shape.insert(out_shape.begin(), 1);
    Tensor<T> grad{ out_shape, 1 };
    grad.non_zero_idxs = utils::total_idxs(out_shape);
    return { grad };
}


// Template declarations

template class Sum<int>;
template class Sum<double>;
template class Sum<long>;
template class Sum<long long>;