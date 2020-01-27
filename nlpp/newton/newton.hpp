#pragma once

#include "newton_dec.hpp"
#include "factorizations.hpp"

namespace nlpp::impl
{

template <class Base_>
template <class Function, class Hessian, class V>
V Newton<Base_>::optimize (const Function& f, const Hessian& hess, V x)
{
    impl::Scalar<V> fx;
    V gx;

    std::tie(fx, gx) = f(x);

    for(int iter = 0; iter < stop.maxIterations(); ++iter)
    {
        auto dir = factorization(gx, hess(x));
    
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
