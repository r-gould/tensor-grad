#ifndef UTILS_TPP
#define UTILS_TPP

#include "utils.hpp"
#include "../tensor.hpp"
#include <iostream>
#include <string>
#include <cassert>

template <class T>
T utils::prod(const std::vector<T>& vec)
{
    T result{ 1 };
    for(T val : vec)
        result *= val;
    return result;
}

template <class T>
void utils::build_str(const Tensor<T>& tensor, std::string& out, std::vector<int> index)
{
    const int index_size{ static_cast<int>(index.size()) };

    if(index_size == tensor.dim)
    {
        const T value{ tensor(index) };
        out += std::to_string(value) + " ";
        return;
    }

    index.push_back(-1);

    out += "[ ";

    for(int i = 0; i < tensor.shape[index_size]; ++i)
    {
        index[index_size] = i;
        utils::build_str(tensor, out, index);
    }

    out += "] ";

    if(index_size == tensor.dim - 1)
        out += '\n';
}

template <class T, class U>
void utils::flatten(const std::vector<U>& vector, std::vector<T>& out)
{
    out.insert(out.end(), vector.begin(), vector.end());
}

template <class T, class U>
void utils::flatten(const std::vector<std::vector<U>>& multidim, std::vector<T>& out)
{
    for (const std::vector<U>& vec : multidim)
        flatten(vec, out);
}

template <class U>
int utils::calc_shape(const std::vector<U>& vector, std::vector<int>& out, bool first_dim)
{
    int curr_size{ static_cast<int>(vector.size()) };

    if(first_dim)
        out.push_back(curr_size);
    
    return curr_size;
}

template <class U>
int utils::calc_shape(const std::vector<std::vector<U>>& multidim, std::vector<int>& out, bool first_dim)
{
    int curr_size{ static_cast<int>(multidim.size()) };
    
    if(first_dim)
        out.push_back(curr_size);

    int inner_size{};
    int prev_size{ static_cast<int>(multidim[0].size()) };

    for (const std::vector<U>& vec : multidim)
    {
        int inner_size = calc_shape(vec, out, first_dim);
        first_dim = false;

        assert((inner_size == prev_size) && "Incorrect size match found in shape.");
        prev_size = inner_size;
    }

    return curr_size;
}


template <class T>
Tensor<T> utils::self_derivative(const std::vector<int>& shape, const bool overwrite_non_zero)
{
    Tensor<T> out{ concat_shapes(shape, shape), 0 };

    if(overwrite_non_zero)
    {
        out.modify([](Tensor<T>& tensor, const std::vector<int>& index)
        {
            std::vector<int> total_index{ concat_shapes(index, index) };
            tensor(total_index) = 1;
            tensor.non_zero_idxs.push_back(total_index);
        }
        , shape);

        return out;

    }

    out.modify([](Tensor<T>& tensor, const std::vector<int>& index)
    {
        tensor(concat_shapes(index, index)) = 1;
    }
    , shape);

    return out;
}

#endif