#ifndef LOG_HPP
#define LOG_HPP

#include "../unary.hpp"
#include "../../../tensor.hpp"
#include <vector>

template <class T>
class Log : public Unary<T>
{
protected:
    const T base;

    Tensor<T>& _forward(Tensor<T>& tensor) override;
    std::vector<Tensor<T>> _backward(Tensor<T>& tensor) override;

public:
    Log(const T base);
};

#endif