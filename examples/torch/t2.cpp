#include <torch/torch.h>
#include "cg/cg.hpp"
#include "TestFunctions/Rosenbrock.h"


torch::Device device = torch::kCUDA;



auto func (torch::Tensor& y)
{
    torch::AutoGradMode guard(true);
    torch::Tensor fx = torch::zeros(1, device);

    for(int i = 0; i < y.size(0) - 1; ++i)
        fx += 100.0 * torch::pow(y[i+1] - torch::pow(y[i], 2), 2) + torch::pow(y[i] - 1.0, 2);

    return fx;
}


template <class V>
auto grad (const Eigen::MatrixBase<V>& x)
{
    torch::Tensor y = torch::ones(x.size(), device);

    for(int i = 0; i < x.rows(); ++i)
        y[i] = x(i);

    y.set_requires_grad(true);

    torch::AutoGradMode guard(false);

    auto fx = func(y);

    fx.backward();
    auto g = y.grad().clone();
    y.grad().zero_();

    nlpp::impl::Plain<V> gx(x.size());

    for(int i = 0; i < x.rows(); ++i)
        gx[i] = g[i].cpu().item().to<double>();

    return gx;
}

auto func (const torch::Tensor& x)
{
    torch::AutoGradMode guard(true);
    return x.norm();
}

auto grad (torch::Tensor& x, torch::Tensor& fx)
{
    fx.backward();
    auto g = x.grad().clone();
    x.grad().zero_();

    return g;
}


int main ()
{
    using nlpp::wrap::Conditions;
    using V = nlpp::Vec;
    using Opt = nlpp::CG<>;

    Opt opt;
    nlpp::Rosenbrock f;
    V x0 = V::Constant(4, 2.0);

    auto g = [](const auto& x) { return grad(x); };

    // auto [x, fx, gx, status] = opt.opt(Opt::functions<V>(func, nlpp::fd::gradient(func)),
    //                                    Opt::domain(x0),
    //                                    Opt::constraints());
    auto [x, fx, gx, status] = opt(f, g, x0, Opt::constraints());

    std::cout << x.transpose() << "\n";

    return 0;
}

