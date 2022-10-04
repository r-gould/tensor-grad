#include "tensor.hpp"
#include "engine/engine.hpp"
#include "utils/utils.hpp"
#include "operations/operation.hpp"
#include "operations/unary/subscript/subscript.hpp"
#include "operations/unary/sum/sum.hpp"
#include "operations/unary/broadcast/broadcast.hpp"
#include "operations/unary/pow/pow.hpp"
#include "operations/unary/exp/exp.hpp"
#include "operations/unary/log/log.hpp"
#include "operations/binary/matmul/matmul.hpp"

#include <cassert>
#include <vector>
#include <string>
#include <functional>
#include <stdexcept>
#include <chrono>

template <class T>
Tensor<T>::Tensor(const std::vector<T> data)
    : data{ data }
    , shape{ std::vector<int>{ static_cast<int>(data.size()) } }
    , dim{ static_cast<int>(shape.size()) }
    , is_scalar{ check_if_scalar() }
{}

template <class T>
Tensor<T>::Tensor(const std::vector<T> data, const std::vector<int> shape)
    : data{ data }
    , shape{ shape }
    , dim{ static_cast<int>(shape.size()) }
    , is_scalar{ check_if_scalar() }
{
    assert(data.size() == utils::prod(shape));
}

template <class T>
Tensor<T>::~Tensor()
{
    /*
    Destructor for Tensor object.
    */

    if(is_accessible && (has_children() || has_parents()))
        ungraph();

    if(grad)
    {
        delete grad;
        grad = nullptr;
    }

    if(oper)
    {
        delete oper;
        oper = nullptr;
    }
}

template <class T>
void Tensor<T>::ungraph()
{
    /*
    Removes tensor from the computational graph, along with its
    children and any associated inaccessibles.
    */

    for(Tensor<T>* parent : parents)
    {
        if(!parent->is_accessible)
        {
            delete parent;
            parent = nullptr;
            continue;
        }

        const auto& iter{ std::find(parent->children.begin(), parent->children.end(), this) };
        
        if(iter != parent->children.end())
            *iter = nullptr;
        else
            throw std::runtime_error("Child tensor not found in parents 'children'.");
    }

    parents.clear();

    for(Tensor<T>* child : children)
        if(child)
        {
            delete child;
            child = nullptr;
        }
    
    children.clear();
}

template <class T>
void Tensor<T>::backprop(std::vector<Tensor<T>*> target_tensors, bool squeeze)
{
    /*
    For each target tensor in 'target_tensors', the derivative
    tensor of *this wrt. each target tensor is computed, and 
    stored in the target tensors 'grad' variable.
    */

    for(Tensor<T>* target : target_tensors)
    {
        if(target->grad)
        {
            delete target->grad;
            target->grad = nullptr;
        }
        
        Tensor<T>* result = new Tensor<T>{ Engine<T>::grad(this, target) };
        if(squeeze)
            *result = Tensor<T>::squeeze(*result);

        target->grad = result;
    }
}

template <class T>
bool Tensor<T>::check_if_scalar() const
{
    return ((this->dim == 1) && (this->shape[0] == 1));
}

template <class T>
Tensor<T> Tensor<T>::squeeze(const Tensor<T>& tensor)
{
    /*
    Removes any dimensions in the shape == 1.
    */

    std::vector<int> shape{ tensor.shape };
    shape.erase(std::remove(shape.begin(), shape.end(), 1), shape.end());
    return Tensor<T>{ tensor.data, shape };
}

template <class T>
void Tensor<T>::add_child(Tensor<T>* child)
{
    this->children.push_back(child);
}

template <class T>
Tensor<T>& Tensor<T>::sum()
{
    /*
    Reduces a tensor to a tensor of shape (1) by summation.
    */

    Sum<T>* oper = new Sum<T>();
    return oper->forward(*this);
}

template <class T>
Tensor<T>& Tensor<T>::pow(const T power)
{
    Pow<T>* oper = new Pow<T>(power);
    return oper->forward(*this);
}

template <class T>
Tensor<T>& Tensor<T>::exp(const T base)
{
    Exp<T>* oper = new Exp<T>(base);
    return oper->forward(*this);
}

template <class T>
Tensor<T>& Tensor<T>::log(const T base)
{
    Log<T>* oper = new Log<T>(base);
    return oper->forward(*this);
}

template <class T>
Tensor<T>& Tensor<T>::index(const std::vector<int>& idx)
{
    Subscript<T>* oper = new Subscript<T>(idx);
    return oper->forward(*this);
}


template <class T>
Tensor<T>& Tensor<T>::broadcast(const std::vector<int>& new_shape)
{
    /*
    Expands a scalar tensor '*this' of shape (1) to a larger tensor 
    filled with 'this->item()' of shape 'new_shape'.
    */

    Broadcast<T>* oper = new Broadcast<T>{ new_shape };
    return oper->forward(*this);
}


template <class T>
std::vector<Tensor<T>*> Tensor<T>::try_broadcast(Tensor<T>& tensor1, Tensor<T>& tensor2)
{
    /*
    Will broadcast if tensor shapes dont match and 
    either is a scalar tensor.
    */

    if(tensor1.shape == tensor2.shape)
        return { &tensor1, &tensor2 };

    else if(tensor1.is_scalar)
        return { &tensor1.broadcast(tensor2.shape), &tensor2 };

    else if(tensor2.is_scalar)
        return { &tensor1, &tensor2.broadcast(tensor1.shape) };

    else
        throw std::runtime_error("Invalid tensor shapes.");
}


template <class T>
bool Tensor<T>::has_parents() const
{
    /*
    Does the tensor have any non-null parents?
    */

    for(const Tensor<T>* parent : parents)
        if(parent)
            return true;

    return false;
}

template <class T>
bool Tensor<T>::has_children() const
{
    /*
    Does the tensor have any non-null children?
    */

    for(const Tensor<T>* child : children)
        if(child)
            return true;

    return false;
}

template <class T>
void Tensor<T>::modify(std::function<void(Tensor<T>&, const std::vector<int>&)> modifier, const std::vector<int>& iter_shape)
{
    /*
    Performs index-wise modification of a tensor, 
    where 'modifier' defines the modification to be done, 
    and 'iter_shape' the index space to iterate over.
    */

    std::vector<std::vector<int>> idxs_set{ utils::total_idxs(iter_shape) };

    for(std::vector<int> idx : idxs_set)
        modifier(*this, idx);
}

template <class T>
void Tensor<T>::set_grad_info(std::vector<Tensor<T>*> parents, Operation<T>* oper)
{
    for(Tensor<T>* parent : parents)
        parent->add_child(this);

    this->parents = parents;
    this->oper = oper;
}

template <class T>
void Tensor<T>::set_accessible_bool(bool is_accessible)
{
    this->is_accessible = is_accessible;
}

template <class T>
T Tensor<T>::item() const
{
    assert(is_scalar);
    return this->data[0];    
}

template <class T>
Tensor<T>& Tensor<T>::matmul(Tensor<T>& other_tensor)
{
    MatMul<T>* oper = new MatMul<T>();
    return oper->forward(*this, other_tensor);
}

template <class T>
Tensor<T>& Tensor<T>::operator- ()
{
    /*
    Unary negation of tensor.
    */

    return (*this) * -1;
}


// Template declarations

template class Tensor<int>;
template class Tensor<double>;
template class Tensor<long>;
template class Tensor<long long>;