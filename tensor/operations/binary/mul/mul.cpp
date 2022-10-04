#include "mul.hpp"
#include "../../../tensor.hpp"
#include <vector>
#include <cassert>

template <class T>
Tensor<T>& Mul<T>::_forward(Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    assert(tensor1.shape == tensor2.shape);

    Tensor<T>* out = new Tensor<T>{ tensor1.shape, 0 };

    out->modify([&tensor1, &tensor2](Tensor<T>& tensor, const std::vector<int>& index)
    {
        tensor(index) = tensor1(index) * tensor2(index);
    }
    , out->shape);
    
    return *out;
}

template <class T>
std::vector<Tensor<T>> Mul<T>::_backward(Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    std::vector<int> grad_shape{ utils::concat_shapes(tensor1.shape, tensor1.shape) };
    Tensor<T> grad1{ grad_shape, 0 };

    grad1.modify([&tensor2](Tensor<T>& tensor, const std::vector<int> index)
    {
        std::vector<int> total_index{ utils::concat_shapes(index, index) };
        tensor(total_index) = tensor2(index);
        tensor.non_zero_idxs.push_back(total_index);
    }
    , tensor1.shape);

    Tensor<T> grad2{ grad_shape, 0 };

    grad2.modify([&tensor1](Tensor<T>& tensor, const std::vector<int> index)
    {
        std::vector<int> total_index{ utils::concat_shapes(index, index) };
        tensor(total_index) = tensor1(index);
        tensor.non_zero_idxs.push_back(total_index);
    }
    , tensor1.shape);

    return { grad1, grad2 };
}


// Template declarations

template class Mul<int>;
template class Mul<double>;
template class Mul<long>;
template class Mul<long long>;