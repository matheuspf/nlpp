#pragma once

#include "GradientDescent_dec.hpp"

namespace nlpp::impl
{

template <class Base_>
template <class Function, class V>
V GradientDescent<Base_>::optimize (const Function& f, V x)
{
    impl::Scalar<V> fx;
    V gx, dir;

    std::tie(fx, gx) = f(x);

    for(int iter = 0; iter < stop.maxIterations(); ++iter)
    {
        dir = -gx;

        auto alpha = lineSearch(f, x, dir);

        x = x + alpha * dir;

        fx = f(x, gx);

        if(stop(*this, x, fx, gx))
            break;

        output(*this, x, fx, gx);
    }

    return x;
}

} // namespace nlpp::impl
