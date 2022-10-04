#include "utils.hpp"
#include <map>

std::vector<std::vector<int>> utils::total_idxs(const std::vector<int>& idx_shape)
{
    static std::map<std::vector<int>, std::vector<std::vector<int>>> cache;
    const auto& iter{ cache.find(idx_shape) };

    if(iter != cache.end())
        return iter->second;

    std::vector<std::vector<int>> idxs_set{};
    bool empty{ true };
    int dims_done{ 0 };
    for(int curr_dim : idx_shape)
    {
        if(empty)
        {
            for(int i = 0; i < curr_dim; ++i)
                idxs_set.push_back( std::vector<int>{ i } );
            
            empty = false;
            ++dims_done;
            continue;
        }

        std::vector<std::vector<int>> new_idxs(idxs_set.size() * curr_dim);
        int count{ 0 };
        for(std::vector<int> idx : idxs_set)
        {
            idx.push_back(-1);
            for(int i = 0; i < curr_dim; ++i)
            {
                idx[dims_done] = i;
                new_idxs[count++] = idx;
            }
        }

        idxs_set = new_idxs;

        ++dims_done;
    }

    cache.insert({idx_shape, idxs_set});

    return idxs_set;
}

std::vector<int> utils::concat_shapes(std::vector<int> shape1, const std::vector<int>& shape2)
{
    shape1.insert(shape1.end(), shape2.begin(), shape2.end());
    return shape1;
}