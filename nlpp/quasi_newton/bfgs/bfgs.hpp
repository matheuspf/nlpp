#pragma once

#include "bfgs_dec.hpp"
#include "initial_hessian.hpp"


namespace nlpp::impl
{

template <class Base>
template <class Function, class V>
V BFGS<Base>::optimize (const Function& f, V x0, InitialHessian initialHessian, LineSearch lineSearch, Stop stop, Output output) const
{
    using Float = impl::Scalar<V>;

    int rows = x0.rows(), cols = x0.cols(), size = rows * cols;

    impl::Plain2D<V> In = impl::Plain2D<V>::Identity(size, size);

    auto hess = initialHessian(f, x0);

    V x1, g0(rows, cols), g1(rows, cols), dir, s, y;

    Float f0 = f(x0, g0);
    Float f1;

    for(int iter = 0; iter < stop.maxIterations; ++iter)
    {
        dir = -hess * g0;

        auto alpha = lineSearch(f, x0, dir);

        x1 = x0 + alpha * dir;

        f1 = f(x1, g1);

        s = x1 - x0;
        y = g1 - g0;

        if(stop(*this, x1, f1, g1))
            break;


        Float rho = 1.0 / std::max(y.dot(s), constants::eps_<Float>);

        hess = (In - rho * s * y.transpose()) * hess * (In - rho * y * s.transpose()) + rho * s * s.transpose();

        x0 = x1;
        g0 = g1;

        output(*this, x1, f1, g1);
    }

    return x1;
}

} // namespace nlpp::impl