#ifndef POW_HPP
#define POW_HPP

#include "../unary.hpp"
#include "../../../tensor.hpp"
#include <vector>

template <class T>
class Pow : public Unary<T>
{
protected:
    const T power;

    Tensor<T>& _forward(Tensor<T>& tensor) override;
    std::vector<Tensor<T>> _backward(Tensor<T>& tensor) override;

public:
    Pow(const T power);
};

#endif