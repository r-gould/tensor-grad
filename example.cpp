#include "tensor/tensor.hpp"
#include "tensor/utils/utils.hpp"
#include <iostream>
#include <vector>
#include <chrono>

// Example usage of Tensor.

int main()
{
    using std::vector;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Can initialize with 1d vector and shape

    Tensor<double> input_a{ 
        {1, 2.5, 3, 4,
         4, 5, 6, 9,
         7, 8, 9, -1,
                        
         9, -1, 3, 1,
         1, 1, 0, 3.9,
         9, 9, 9, 0},               
    
        {2, 3, 4} };

    // Or directly from a multidimensional vector

    Tensor<double> input_b{ vector<vector<vector<double>>> {
        {{3, 7, -3, -2},
         {-7, -5, 1, 0},
         {2, 0, 0, -9}},
                                                    
        {{2, -5, 51, 23},
         {1, 0, 0, -12},
         {-1, -2, -3, 0}}
    } };

    Tensor<double> input_c{ vector<vector<double>> {
        {12, -8, 6},
        {-7, 98, 8},
        {1, 4, -2},
        {4, -4, 1},
        {12, -9, -8}
    } };

    Tensor<double> weight{ vector<vector<double>> {
        {0.1, 0.2, 0.3, 0.4, -0.5},
        {0.7, -0.9, -0.2, -0.11, 0.32},
        {0.12, 0.24, 0.432, 0.45, 0.34}
    } };

    Tensor<double> bias{ vector<vector<double>> {
        {-2, 12, 3},
        {-7, -8, 0},
        {0, 3, -1}
    } };

    // Forward computations

    Tensor<double>& out_a = input_a * input_a + input_b - 1;

    Tensor<double>& out_b = 2*out_a + 6 + input_a + input_b * input_b - input_b.sum();

    Tensor<double>& out_c = (weight*weight).log() + (weight.matmul(input_c).pow(2) + bias).matmul(weight.exp(2)) - weight.sum() + 3*bias.sum() - 4;

    Tensor<double>& loss = out_b.sum() + input_a.sum() - out_c.sum() + weight.exp(2).sum();

    // Backprop

    // Compute derivative of loss wrt. input_a, input_b, weight, and bias
    loss.backprop({ &input_a, &input_b, &weight, &bias });


    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end_time - start_time;
    std::cout << "Done in " << diff.count() << "ms\n";


    std::cout << "Derivative of loss wrt.:" << '\n';

    std::cout << "input_a: " << *(input_a.grad) << '\n';
    std::cout << "input_b: " << *(input_b.grad) << '\n';
    std::cout << "weight: " << *(weight.grad) << '\n';
    std::cout << "bias: " << *(bias.grad) << '\n';

    int i{};
    std::cin >> i;

    return 0;
}