#pragma once

#include "lbfgs_dec.hpp"
#include "../bfgs/initial_hessian.hpp"

namespace nlpp::impl
{

template <class Base>
template <class Function, class V>
V LBFGS<Base>::optimize (const Function& f, V x0, InitialHessian initialHessian, LineSearch lineSearch, Stop stop, Output output) const
{
    using Float = impl::Scalar<V>;

    V gx(x0.rows()), gx0(x0.rows());
    Float fx0, fx;

    fx0 = f(x0, gx0);

    std::deque<V> vs;
    std::deque<V> vy;

    for(int iter = 0; iter < stop.maxIterations; ++iter)
    {
        auto H = initialHessian(f, x0);

        V p = direction(f, x0, gx0, H, vs, vy);

        auto alpha = lineSearch(f, x0, p);

        V x = x0 + alpha * p;

        fx = f(x, gx);

        if(stop(*this, x, fx, gx))
            return x;

        auto s = x - x0;
        auto y = gx - gx0;

        vs.push_back(s);
        vy.push_back(y);

        if(iter > std::min(m, (int)x0.size()))
        {
            vs.pop_front();
            vy.pop_front();
        }


        std::tie(x0, fx0, gx0) = std::tie(x, fx, gx);

        output(*this, x, fx, gx);
    }

    return x0;
}


template <class Base>
template <class Function, class V, class U>
V LBFGS<Base>::direction (const Function& f, const V& x, const V& gx, const Eigen::MatrixBase<U>& H, const std::deque<V>& vs, const std::deque<V>& vy) const
{
    using W = nlpp::VecX<Scalar<V>>;

    W alpha(vs.size());
    W rho(vs.size());
    W q = -gx;

    for(int i = vs.size() - 1; i >= 0; --i)
    {
        rho[i] = 1.0 / vy[i].dot(vs[i]);
        alpha[i] = rho[i] * vs[i].dot(q);

        q = q - alpha[i] * vy[i];
    }

    W r = H * q;

    for(int i = 0; i < vs.size(); ++i)
    {
        auto beta = rho[i] * vy[i].dot(r);

        r = r + (alpha[i] - beta) * vs[i];
    }

    return r;
}


} // namespace nlpp::impl
