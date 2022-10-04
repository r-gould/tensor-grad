#ifndef UTILS_HPP
#define UTILS_HPP

template <class T>
class Tensor;

#include <vector>
#include <string>

namespace utils
{
    template <class T>
    T prod(const std::vector<T>& vec);

    template <class T>
    void build_str(const Tensor<T>& tensor, std::string& out, std::vector<int> index = {});

    template <class T, class U>
    void flatten(const std::vector<U>& vector, std::vector<T>& out);

    template <class T, class U>
    void flatten(const std::vector<std::vector<U>>& multidim, std::vector<T>& out);

    template <class U>
    int calc_shape(const std::vector<U>& vector, std::vector<int>& out, bool first_dim = true);

    template <class U>
    int calc_shape(const std::vector<std::vector<U>>& multidim, std::vector<int>& out, bool first_dim = true);

    std::vector<std::vector<int>> total_idxs(const std::vector<int>& idx_shape);

    std::vector<int> concat_shapes(std::vector<int> shape1, const std::vector<int>& shape2);

    template <class T>
    Tensor<T> self_derivative(const std::vector<int>& shape, const bool overwrite_non_zero = false);
}

#include "utils.tpp"

#endif