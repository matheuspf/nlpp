#include <torch/torch.h>
#include "cg/cg.hpp"
#include "TestFunctions/Rosenbrock.h"


torch::Device device = torch::kCPU;

torch::Tensor fromEigen (const nlpp::impl::VecType auto& x)
{
    torch::Tensor y = torch::zeros(x.size(), device);

    for(int i = 0; i < y.size(0); ++i)
        y[i] = x(i);
    
    return y;
}

template <typename F=double> nlpp::impl::VecType auto fromTorch (const torch::Tensor& x)
{
    nlpp::VecX<F> y(x.size(0));

    for(int i = 0; i < y.size(); ++i)
        y(i) = x[i].item().to<double>();
    
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

    double fxi = fx.item().to<double>();

    return std::make_tuple(fxi, gx);
}

template <class> class Prt;


int main ()
{
    using Opt = nlpp::CG<>;
    using V = nlpp::Vec;

    auto ff = [](const auto& x){ return funcGrad(x); };


    // auto ff = [](const auto& x) -> double{ return 10.0; };
    // auto ff = [](const auto& x){ return 10.0; };


    // auto ff = [](const nlpp::Vec& x) -> std::tuple<double, nlpp::Vec> { return funcGrad2(x); };

    // auto f = nlpp::wrap::funcGrad(ff);


    static constexpr nlpp::wrap::Conditions conditions = nlpp::traits::Optimizer<Opt>::conditions;

    auto xx = nlpp::wrap::fd::functions<conditions, V>(ff, [](const auto& x){ return 10.0; });


    // nlpp::Rosenbrock func;

    // Opt opt;
    // V x0 = V::Constant(5, 2.0);

    // auto [x, fx, gx, status] = opt(ff, x0);


    return 0;
}
