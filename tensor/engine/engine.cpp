#include "engine.hpp"
#include "../utils/utils.hpp"
#include <vector>
#include <algorithm>
#include <map>

template <class T>
Tensor<T> Engine<T>::grad(Tensor<T>* node, Tensor<T>* target)
{
    /*
    Computes the derivative of node with respect to target.
    */

    if(node == target)
        return utils::self_derivative<T>(node->shape, true);

    else if(!node->has_parents())
        return Tensor<T>{ { 0 } };

    std::vector<Tensor<T>*> parents{ node->parents };
    std::vector<Tensor<T>> node_wrt_parents{ node->oper->backward(parents) };
    
    const std::vector<int> node_target_shape{ utils::concat_shapes(node->shape, target->shape) };
    Tensor<T> node_wrt_target{ node_target_shape, 0 };

    int i{ 0 };
    for(Tensor<T>* parent : parents)
    {
        Tensor<T> node_wrt_parent = node_wrt_parents[i++];
        Tensor<T> parent_wrt_target = grad(parent, target);
        const int parent_dim{ parent->dim };
        
        update(node_wrt_target, node_wrt_parent, parent_wrt_target, parent_dim);
    }

    return node_wrt_target;
}

template <class T>
void Engine<T>::update(Tensor<T>& node_wrt_target, const Tensor<T>& node_wrt_parent, const Tensor<T>& parent_wrt_target, const int parent_dim)
{
    if(parent_wrt_target.non_zero_idxs.size() == 0)
        return;

    std::map<std::vector<int>, std::vector<int>> cache;

    bool first_run{ true };
    for(std::vector<int> n_wrt_p_idx : node_wrt_parent.non_zero_idxs)
    {
        if(first_run)
        {
            first_update(cache, node_wrt_target, node_wrt_parent, parent_wrt_target, n_wrt_p_idx, parent_dim);
            first_run = false;
            continue;
        }

        cached_update(cache, node_wrt_target, node_wrt_parent, parent_wrt_target, n_wrt_p_idx, parent_dim);
    }
}

template <class T>
void Engine<T>::first_update(std::map<std::vector<int>, std::vector<int>>& cache, Tensor<T>& node_wrt_target, const Tensor<T>& node_wrt_parent, const Tensor<T>& parent_wrt_target, const std::vector<int>& n_wrt_p_idx, const int parent_dim)
{
    int node_dim{ node_wrt_parent.dim - parent_dim };
    int cache_index{ 0 };

    for(std::vector<int> p_wrt_t_idx : parent_wrt_target.non_zero_idxs)
    {
        std::vector<int> p_idx{};

        bool is_valid{ true };
        for(int i = 0; i < parent_dim; ++i)
        {
            if(n_wrt_p_idx[node_dim+i] != p_wrt_t_idx[i])
                is_valid = false;

            p_idx.push_back(p_wrt_t_idx[i]);
        }

        const auto& iter{ cache.find(p_idx) };

        if(iter != cache.end())
            iter->second.push_back(cache_index);
        
        else
            cache.insert({p_idx, { cache_index }});

        ++cache_index;


        if(!is_valid)
            continue;

        std::vector<int> n_wrt_t_idx{ get_n_wrt_t_idx(n_wrt_p_idx, p_wrt_t_idx, parent_dim) };

        node_wrt_target(n_wrt_t_idx) += node_wrt_parent(n_wrt_p_idx) * parent_wrt_target(p_wrt_t_idx);
        
        if(std::find(node_wrt_target.non_zero_idxs.begin(), node_wrt_target.non_zero_idxs.end(), n_wrt_t_idx) == node_wrt_target.non_zero_idxs.end())
            node_wrt_target.non_zero_idxs.push_back(n_wrt_t_idx);
    }
}

template <class T>
void Engine<T>::cached_update(const std::map<std::vector<int>, std::vector<int>>& cache, Tensor<T>& node_wrt_target, const Tensor<T>& node_wrt_parent, const Tensor<T>& parent_wrt_target, const std::vector<int>& n_wrt_p_idx, const int parent_dim)
{
    int node_dim{ node_wrt_parent.dim - parent_dim };
    std::vector<int> p_idx{};

    for(int i = 0; i < parent_dim; ++i)
        p_idx.push_back(n_wrt_p_idx[node_dim+i]);

    const auto& iter{ cache.find(p_idx) };

    if(iter == cache.end())
        return;

    for(int i : iter->second)
    {
        std::vector<int> p_wrt_t_idx{ parent_wrt_target.non_zero_idxs[i] };
        std::vector<int> n_wrt_t_idx{ get_n_wrt_t_idx(n_wrt_p_idx, p_wrt_t_idx, parent_dim) };
    
        node_wrt_target(n_wrt_t_idx) += node_wrt_parent(n_wrt_p_idx) * parent_wrt_target(p_wrt_t_idx);
    
        if(std::find(node_wrt_target.non_zero_idxs.begin(), node_wrt_target.non_zero_idxs.end(), n_wrt_t_idx) == node_wrt_target.non_zero_idxs.end())
            node_wrt_target.non_zero_idxs.push_back(n_wrt_t_idx);
    }
}

template <class T>
std::vector<int> Engine<T>::get_n_wrt_t_idx(const std::vector<int>& n_wrt_p_idx, const std::vector<int>& p_wrt_t_idx, const int parent_dim)
{
    const int node_dim{ static_cast<int>(n_wrt_p_idx.size()) - parent_dim };
    std::vector<int> n_wrt_t_idx{};

    for(int i = 0; i < node_dim; ++i)
        n_wrt_t_idx.push_back(n_wrt_p_idx[i]);

    for(int i = parent_dim; i < p_wrt_t_idx.size(); ++i)
        n_wrt_t_idx.push_back(p_wrt_t_idx[i]);

    return n_wrt_t_idx;
}

// Template declarations

template class Engine<int>;
template class Engine<double>;
template class Engine<long>;
template class Engine<long long>;