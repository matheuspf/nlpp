#pragma once

#include "newton_dec.hpp"
#include "factorizations.hpp"

namespace nlpp::impl
{

template <class Base_>
template <class Function, class V>
V Newton<Base_>::optimize (const Function& f, V x, Factorization factorization, LineSearch lineSearch, Stop stop, Output out) const
{
    auto [fx, gx] = f(x);

    for(int iter = 0; iter < stop.maxIterations; ++iter)
    {
        auto dir = factorization(gx, f.hessian(x));
    
        auto alpha = lineSearch(f, x, dir);
    
        x = x + alpha * dir;
    
        fx = f(x, gx);
    
    
        if(stop(*this, x, fx, gx))
            break;
    
        out(*this, x, fx, gx);
    }

    return x;
}


} // namespace nlpp::impl
