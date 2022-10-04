#include "subscript.hpp"
#include "../../../utils/utils.hpp"
#include "../../../tensor.hpp"
#include <cassert>
#include <vector>

template <class T>
Subscript<T>::Subscript(const std::vector<int>& index)
    : index{ index }
    , index_size{ static_cast<int>(index.size()) }
{}

template <class T>
Tensor<T>& Subscript<T>::_forward(Tensor<T>& tensor)
{
    assert(tensor.dim == this->index_size);
    
    int flat_index{ tensor.flatten_index(this->index) };
    std::vector<T> scalar{ tensor.data[flat_index] };
    
    Tensor<T>* out = new Tensor<T>{ scalar };
    return *out;
}

template <class T>
std::vector<Tensor<T>> Subscript<T>::_backward(Tensor<T>& tensor)
{
    std::vector<int> grad_shape{ tensor.shape };
    grad_shape.insert(grad_shape.begin(), 1);
    
    Tensor<T> grad{ grad_shape, 0 };

    std::vector<int> total_index{ this->index };
    total_index.insert(total_index.begin(), 0);

    grad(total_index) = 1;
    grad.non_zero_idxs = { total_index };
    return { grad };
}


// Template declarations

template class Subscript<int>;
template class Subscript<double>;
template class Subscript<long>;
template class Subscript<long long>;