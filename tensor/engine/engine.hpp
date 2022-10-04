#ifndef ENGINE_HPP
#define ENGINE_HPP

#include "../tensor.hpp"
#include <map>

template <class T>
class Engine
{
public:
    static Tensor<T> grad(Tensor<T>* node, Tensor<T>* target);
    
private:
    static void update(Tensor<T>& node_wrt_target, const Tensor<T>& node_wrt_parent, const Tensor<T>& parent_wrt_target, const int parent_dim);
    static void first_update(std::map<std::vector<int>, std::vector<int>>& cache, Tensor<T>& node_wrt_target, const Tensor<T>& node_wrt_parent, const Tensor<T>& parent_wrt_target, const std::vector<int>& n_wrt_p_idx, const int parent_dim);
    static void cached_update(const std::map<std::vector<int>, std::vector<int>>& cache, Tensor<T>& node_wrt_target, const Tensor<T>& node_wrt_parent, const Tensor<T>& parent_wrt_target, const std::vector<int>& n_wrt_p_idx, const int parent_dim);
    
    static std::vector<int> get_n_wrt_t_idx(const std::vector<int>& n_wrt_p_idx, const std::vector<int>& p_wrt_t_idx, const int parent_dim);
};

#endif