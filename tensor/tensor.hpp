#ifndef TENSOR_HPP
#define TENSOR_HPP

template <class T>
class Operation;

template <class T>
class Subscript;

template <class T>
class Sum;

template <class T>
class Engine;

#include "operations/operation.hpp"
#include "operations/unary/subscript/subscript.hpp"
#include "operations/unary/sum/sum.hpp"
#include <iostream>
#include <vector>
#include <functional>
#include <initializer_list>
#include <cmath>

template <class T>
class Tensor
{
private:
    /*
    Underlying representation of the tensor. Stored in 1-dimensional
    array (row-major).
    */

    std::vector<T> data;

    /*
    During the forward process, some tensors are created that 
    are inaccessible to the user, so to avoid a memory leak, these 
    are flagged with 'is_accessible = false' such that they are 
    properly destructed.
    */

    bool is_accessible{ true };

    /*
    Tensors responsible for the creation of 'this' are stored 
    in 'parents'.

    Tensors produced from 'this' are stored in 'children'.

    'oper' is the operation that was applied to 'parents' to 
    create 'this'.
    */

    std::vector<Tensor<T>*> parents;
    std::vector<Tensor<T>*> children;
    Operation<T>* oper = nullptr;

    bool check_if_scalar() const;

    void add_child(Tensor<T>* child);

    void set_accessible_bool(bool is_accessible);

    Tensor<T>& broadcast(const std::vector<int>& new_shape);

    static std::vector<Tensor<T>*> try_broadcast(Tensor<T>& tensor1, Tensor<T>& tensor2);

public:
    // Public variables

    std::vector<int> shape;
    int dim;
    bool is_scalar;

    Tensor<T>* grad = nullptr;

    std::vector<std::vector<int>> non_zero_idxs{};

    // Constructors and destructor

    Tensor();
    Tensor(const std::vector<T> data);
    Tensor(const std::vector<T> data, const std::vector<int> shape);

    template <class U>
    Tensor(const std::vector<std::vector<U>>& data);

    template <class U>
    Tensor(const std::vector<int> shape, const U value);

    ~Tensor();

    //

    void modify(std::function<void(Tensor<T>&, const std::vector<int>&)> modifier, const std::vector<int>& iter_shape);
    
    template <class Container>
    int flatten_index(const Container& indices) const;

    template <class U>
    static std::vector<T> flatten(const std::vector<std::vector<U>>& data);

    template <class U>
    static std::vector<int> calc_shape(const std::vector<std::vector<U>>& data);

    static Tensor<T> squeeze(const Tensor<T>& tensor);

    void backprop(std::vector<Tensor<T>*> target_tensors, bool squeeze = true);

    void ungraph();

    void set_grad_info(std::vector<Tensor<T>*> parents, Operation<T>* op);
    
    bool has_parents() const;

    bool has_children() const;

    T item() const;

    // Custom operations

    Tensor<T>& sum();

    Tensor<T>& pow(const T power);

    Tensor<T>& exp(const T base = std::exp(1.0));

    Tensor<T>& log(const T base = std::exp(1.0));

    Tensor<T>& index(const std::vector<int>& idx);

    Tensor<T>& matmul(Tensor<T>& other_tensor);


    
    // Operator overloads

    template <class... A>
    T& operator() (A... args);

    template <class Container>
    T& operator() (const Container& indices);

    template <class Container>
    const T operator() (const Container& indices) const;

    template <class U>
    friend Tensor<U>& operator+ (Tensor<U>& tensor1, Tensor<U>& tensor2);
    
    template <class U, class V>
    friend Tensor<U>& operator+ (Tensor<U>& tensor, const V value);

    template <class U, class V>
    friend Tensor<U>& operator+ (const V value, Tensor<U>& tensor);


    template <class U>
    friend Tensor<U>& operator- (Tensor<U>& tensor1, Tensor<U>& tensor2);

    template <class U, class V>
    friend Tensor<U>& operator- (Tensor<U>& tensor, const V value);

    template <class U, class V>
    friend Tensor<U>& operator- (const V value, Tensor<U>& tensor);


    template <class U>
    friend Tensor<U>& operator* (Tensor<U>& tensor1, Tensor<U>& tensor2);

    template <class U, class V>
    friend Tensor<U>& operator* (Tensor<U>& tensor, const V value);

    template <class U, class V>
    friend Tensor<U>& operator* (const V value, Tensor<U>& tensor);

    template <class U>
    friend Tensor<U>& operator/ (Tensor<U>& tensor1, Tensor<U>& tensor2);

    template <class U, class V>
    friend Tensor<U>& operator/ (Tensor<U>& tensor, const V value);

    template <class U, class V>
    friend Tensor<U>& operator/ (const V value, Tensor<U>& tensor);

    Tensor<T>& operator- ();

    template <class U>
    friend std::ostream& operator<< (std::ostream& out, const Tensor<U>& tensor);

    // Friend classes

    friend class Subscript<T>;
    
    friend class Sum<T>;

    friend class Engine<T>;
};


#include "tensor.tpp"


#endif