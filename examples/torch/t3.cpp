#include <torch/torch.h>
#include "cg/cg.hpp"
#include "TestFunctions/Rosenbrock.h"


torch::Device device = torch::kCUDA;


torch::Tensor fromEigen (const nlpp::impl::VecType auto& x)
{
    torch::Tensor y = torch::zeros(x.size(), device);

    for(int i = 0; i < x.size(); ++i)
        y[i] = x(i);
    
    return y;
}

template <typename F=double>
nlpp::impl::VecType auto fromTorch (const torch::Tensor& x)
{
    nlpp::VecX<F> y(x.size);

    for(int i = 0; i < x.size(); ++i)
        y(i) = x[i];
    
    return y;
}


torch::Tensor func (torch::Tensor& x)
{
    torch::Tensor fx = torch::zeros(1, device);

    for(int i = 0; i < x.size(0) - 1; ++i)
        fx += 100.0 * torch::pow(x[i+1] - torch::pow(x[i], 2), 2) + torch::pow(x[i] - 1.0, 2);

    return fx;
}


auto funcGrad(const nlpp::impl::VecType auto& x)
{
    torch::Tensor y = fromEigen(x);
    y.set_requires_grad(true);

    torch::AutoGradMode guard(true);

    torch::Tensor fx = func(y);
    fx.backward();

    auto gy = y.grad().clone();
    y.grad().zero_();

    auto gx = fromTorch(gy);

    return std::make_tuple(fx.item().to<double>(), gx);
}


int main ()
{
    using Opt = nlpp::CG<>;
    using V = nlpp::Vec;

    auto ff = [](const auto& x){ return funcGrad(x); };

    // auto f = nlpp::wrap::funcGrad(ff);

    Opt opt;
    V x0 = V::Constant(5, 2.0);

    auto [x, fx, gx, status] = opt(ff, x0);

    return 0;
}