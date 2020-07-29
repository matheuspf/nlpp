#pragma once

#include "../helpers/helpers.hpp"


namespace nlpp
{

template <class Impl, typename Float = types::Float>
struct P95_98
{
    template <class V>
    auto function (const V& x) const
    {
        return 4.3*x(0) + 31.8*x(1) + 63.3*x(2) + 15.8*x(3) + 68.5*x(4) + 4.7*x(5);
    }

    template <class V>
    auto ineqs (const V& x) const
    {
        VecX<impl::Scalar<V>> g(4);
        
        g(0) = -(17.1*x(0) + 38.2*x(1) + 204.2*x(2) + 212.3*x(3) + 623.4*x(4) + 1495.5*x(5) -
                 169*x(0)*x(2) - 3580*x(2)*x(4) - 3810*x(3)*x(4) - 18500*x(3)*x(5) - 24300*x(4)*x(5) - Impl::b(0));
        g(1) = -(17.9*x(0) + 36.8*x(1) + 113.9*x(2) + 169.7*x(3) + 337.8*x(4) + 1385.2*x(5)
                 -139*x(0)*x(2) - 2450*x(3)*x(4) - 16600*x(3)*x(5) - 17200*x(4)*x(5) - Impl::b(1));
        g(2) = -(273*x(1) - 70*x(3) - 819*x(4) + 26000*x(3)*x(4) - Impl::b(2));
        g(3) = -(59.9*x(0) - 311*x(1) + 587*x(3) + 391*x(4) + 2198*x(5) - 14000*x(0)*x(5) - Impl::b(3));

        return g;
    }

    static const VecX<Float, 6> lb;
    static const VecX<Float, 6> ub;
};

template <class Impl, typename Float>
const VecX<Float, 6> P95_98<Impl, Float>::lb = {0.0,  0.0,   0.0,   0.0,   0.0,   0.0};

template <class Impl, typename Float>
const VecX<Float, 6> P95_98<Impl, Float>::ub = {0.31, 0.046, 0.068, 0.042, 0.028, 0.0134};


template <typename Float = types::Float>
struct P95 : public P95_98<P95<Float>, Float>
{
    static const VecX<Float, 4> b;
};

template <typename Float>
const VecX<Float, 4> P95<Float>::b = {4.97, -1.88, -29.08, -78.02};


} // namespace nlpp
