#ifndef EXP_HPP
#define EXP_HPP

#include "../unary.hpp"
#include "../../../tensor.hpp"
#include <vector>

template <class T>
class Exp : public Unary<T>
{
protected:
    const T base;

    Tensor<T>& _forward(Tensor<T>& tensor) override;
    std::vector<Tensor<T>> _backward(Tensor<T>& tensor) override;

public:
    Exp(const T base);
};

#endif