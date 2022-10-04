#ifndef TENSOR_TPP
#define TENSOR_TPP

template <class T>
class Add;

template <class T>
class Mul;

#include "tensor.hpp"
#include "utils/utils.hpp"
#include "operations/binary/add/add.hpp"
#include "operations/binary/mul/mul.hpp"
#include <cassert>
#include <string>


template <class T>
template <class U>
Tensor<T>::Tensor(const std::vector<std::vector<U>>& data)
    : data{ flatten(data) }
    , shape{ calc_shape(data) }
    , dim{ static_cast<int>(shape.size()) }
    , is_scalar{ check_if_scalar() }
{}

template <class T>
template <class U>
Tensor<T>::Tensor(const std::vector<int> shape, const U value)
    : shape{ shape }
    , dim{ static_cast<int>(shape.size()) }
    , is_scalar{ check_if_scalar() }
{
    T casted_value{ static_cast<T>(value) };
    int num_values{ utils::prod(shape) };
    const std::vector<T> data(num_values, casted_value);

    this->data = data;
}



template <class T>
template <class Container>
int Tensor<T>::flatten_index(const Container& indices) const
{
    /*
    Takes a 'dim'-dimensional index and flattens it to a 
    1-dimensional index to fetch its associated value in this->data.
    */

    assert(indices.size() == dim);

    int flat_index{ 0 };
    int dim_coeff{ 1 };
    for(int i = 0; i < dim; ++i)
    {
        assert((indices[i] >= 0) && (indices[i] < shape[i]));
        flat_index += indices[dim-i-1] * dim_coeff;
        dim_coeff *= shape[dim-i-1];
    }

    return flat_index;
}


template <class T>
template <class U>
std::vector<T> Tensor<T>::flatten(const std::vector<std::vector<U>>& data)
{
    std::vector<T> flat_data{};
    utils::flatten(data, flat_data);
    return flat_data;
}

template <class T>
template <class U>
std::vector<int> Tensor<T>::calc_shape(const std::vector<std::vector<U>>& data)
{
    std::vector<int> shape{};
    utils::calc_shape(data, shape);
    return shape;
}




template <class T>
template <class Container>
T& Tensor<T>::operator() (const Container& indices)
{
    return this->data[ flatten_index(indices) ];
}

template <class T>
template <class Container>
const T Tensor<T>::operator() (const Container& indices) const
{
    return this->data[ flatten_index(indices) ];
}

template <class T>
template <class... A>
T& Tensor<T>::operator() (A... indices)
{
    return (*this)(std::vector<int>{ indices... });
}




template <class U>
std::ostream& operator<< (std::ostream& out, const Tensor<U>& tensor)
{
    std::string out_str{};
    utils::build_str(tensor, out_str);
    return (out << "Tensor" << out_str);
}



template <class T>
Tensor<T>& operator+ (Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    /*
    Elementwise tensor addition.
    */

    Add<T>* oper = new Add<T>();
    return oper->forward( Tensor<T>::try_broadcast(tensor1, tensor2) );
}

template <class T, class U>
Tensor<T>& operator+ (Tensor<T>& tensor, const U value)
{
    Tensor<T>* value_tensor = new Tensor<T>{ tensor.shape, value };
    value_tensor->set_accessible_bool(false);

    return tensor + (*value_tensor);
}

template <class T, class U>
Tensor<T>& operator+ (const U value, Tensor<T>& tensor)
{
    return tensor + value;
}




template <class T>
Tensor<T>& operator- (Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    /*
    Elementwise tensor subtraction.
    */

    return tensor1 + (-tensor2);
}

template <class T, class U>
Tensor<T>& operator- (Tensor<T>& tensor, const U value)
{
    return tensor + (-value);
}

template <class T, class U>
Tensor<T>& operator- (const U value, Tensor<T>& tensor)
{
    return value + (-tensor);
}




template <class T>
Tensor<T>& operator* (Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    /*
    Elementwise tensor multiplication.
    */

    Mul<T>* oper = new Mul<T>();
    return oper->forward( Tensor<T>::try_broadcast(tensor1, tensor2) );
}

template <class T, class U>
Tensor<T>& operator* (Tensor<T>& tensor, const U value)
{
    Tensor<T>* value_tensor = new Tensor<T>{ tensor.shape, value };
    value_tensor->set_accessible_bool(false);

    return tensor * (*value_tensor);
}

template <class T, class U>
Tensor<T>& operator* (const U value, Tensor<T>& tensor)
{
    return tensor * value;
}





template <class T>
Tensor<T>& operator/ (Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    /*
    Elementwise tensor division.
    */

    return tensor1 * tensor2.pow(-1);
}

template <class T, class U>
Tensor<T>& operator/ (Tensor<T>& tensor, const U value)
{
    return tensor * (1.0 / value);
}

template <class T, class U>
Tensor<T>& operator/ (const U value, Tensor<T>& tensor)
{
    return value * tensor.pow(-1);
}



#endif