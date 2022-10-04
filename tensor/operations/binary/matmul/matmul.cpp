#include "matmul.hpp"
#include "../../../tensor.hpp"
#include <vector>
#include <cassert>

template <class T>
Tensor<T>& MatMul<T>::_forward(Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    assert((tensor1.dim == 2) && (tensor2.dim == 2));

    int rows1{ tensor1.shape[0] }, cols1{ tensor1.shape[1] };
    int rows2{ tensor2.shape[0] }, cols2{ tensor2.shape[1] };

    assert(cols1 == rows2);

    Tensor<T>* out = new Tensor<T>{ { rows1, cols2 }, 0 };
    
    out->modify([&tensor1, &tensor2](Tensor<T>& tensor, const std::vector<int>& index)
    {
        tensor(index[0], index[2]) += tensor1(index[0], index[1]) * tensor2(index[1], index[2]);
    }
    , { rows1, cols1, cols2 });

    return *out;
}

template <class T>
std::vector<Tensor<T>> MatMul<T>::_backward(Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    int rows1{ tensor1.shape[0] }, cols1{ tensor1.shape[1] };
    int rows2{ tensor2.shape[0] }, cols2{ tensor2.shape[1] };

    Tensor<T> grad1{ { rows1, cols2, rows1, cols1 }, 0 };

    grad1.modify([&tensor2](Tensor<T>& tensor, const std::vector<int>& index)
    {
        std::vector<int> total_index{ index[0], index[1], index[0], index[2] };
        tensor(total_index) = tensor2(index[2], index[1]);
        tensor.non_zero_idxs.push_back(total_index);
    }
    , { rows1, cols2, cols1 });

    Tensor<T> grad2{ { rows1, cols2, rows2, cols2 }, 0 };

    grad2.modify([&tensor1](Tensor<T>& tensor, const std::vector<int>& index)
    {
        std::vector<int> total_index{ index[0], index[1], index[2], index[1] };
        tensor(total_index) = tensor1(index[0], index[2]);
        tensor.non_zero_idxs.push_back(total_index);
    }
    , { rows1, cols2, rows2 });

    return { grad1, grad2 };
}


// Template declarations

template class MatMul<int>;
template class MatMul<double>;
template class MatMul<long>;
template class MatMul<long long>;