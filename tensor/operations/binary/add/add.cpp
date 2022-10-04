#include "add.hpp"
#include "../../../utils/utils.hpp"
#include "../../../tensor.hpp"
#include <cassert>
#include <vector>

template <class T>
Tensor<T>& Add<T>::_forward(Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    assert(tensor1.shape == tensor2.shape);

    Tensor<T>* out = new Tensor<T>{ tensor1.shape, 0 };

    out->modify([&tensor1, &tensor2](Tensor<T>& tensor, const std::vector<int>& index)
    {
        tensor(index) = tensor1(index) + tensor2(index);
    }
    , out->shape);

    return *out;
}

template <class T>
std::vector<Tensor<T>> Add<T>::_backward(Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    Tensor<T> grad{ utils::self_derivative<T>(tensor1.shape, true) };
    return { grad, grad };
}

// Template initialization

template class Add<int>;
template class Add<double>;
template class Add<long>;
template class Add<long long>;