#pragma once

#include "../helpers/helpers.hpp"


namespace nlpp
{

using impl::VecType;

struct TreeBarTrussFunc
{
    template <class V>
    impl::Scalar<V> operator() (const V& x) const
    {
        return (2.0 * std::sqrt(2.0) * x(0) + x(1)) * 100.0;
    }
};

struct TreeBarTrussIneqs
{
    template <class V>
    VecX<impl::Scalar<V>> operator() (const V& x) const
    {
        using Float = impl::Scalar<V>;

        // VecX<Float, 7> r;
        VecX<Float> r(7);

        r(0) = -x(0);
        r(1) = -x(1);
        r(2) = x(0) - 1.0;
        r(3) = x(1) - 1.0;

        Float l = 100.0;
        Float P = 2.0;
        Float sig = 2.0;

        r(4) = ((std::sqrt(2) * x(0) + x(1)) / (std::sqrt(2) * std::pow(x(0), 2) + 2 * x(0) * x(1))) * P - sig;

        r(5) = (x(1) / (std::sqrt(2) * std::pow(x(0), 2) + 2 * x(0) * x(1))) * P - sig;

        r(6) = (1.0 / (x(0) + std::sqrt(2) * x(1))) * P - sig;

        return r;
    }
};

} // namespace nlpp