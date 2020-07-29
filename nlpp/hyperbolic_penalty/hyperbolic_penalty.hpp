#pragma once

#include "hyperbolic_penalty_dec.hpp"
#include "../quasi_newton/lbfgs/lbfgs.hpp"
#include "../utils/finite_difference_dec.hpp"


namespace nlpp::impl
{

template <class Base_>
template <class Function, class V, class Inequalities>
V HyperbolicPenalty<Base_>::optimize (const Function& function, V x, const Inequalities& inequalities, Stop stop, Output output) const
{
    Float lambda = lambda0, tau = tau0;

    auto penalty = ::nlpp::wrap::fd::funcGrad([&](const V& x) -> Scalar<V>
    {
        auto ineqs = inequalities(x);
        return (lambda * ineqs.array() + Eigen::sqrt(std::pow(lambda, 2) * Eigen::pow(-ineqs.array(), 2) + std::pow(tau, 2))).sum();
    });

    auto optFunc = [&](const V& x) -> Scalar<V>
    {
        return function.function(x) + penalty.function(x);
    };

    auto optGrad = [&](const V& x) -> V
    {
        return function.gradient(x) + penalty.gradient(x);
    };

    for(int iter = 0; iter < stop.maxIterations; ++iter)
    {
        x = optimizer(optFunc, optGrad, x);

        std::cout << "Grad:\t" << optGrad(x).transpose() << "\t" << optGrad(x).norm() << "\n";

        if(optGrad(x).norm() < stop.gTol)
            break;

        if((inequalities(x).array() > 0.0).any())
            lambda = r * lambda;

        else
            tau = q * tau;
    }

    return x;
}

} // namespace nlpp::impl
