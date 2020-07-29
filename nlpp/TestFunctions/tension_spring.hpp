#pragma once

#include "../helpers/helpers.hpp"


namespace nlpp
{

struct TensionSpring
{
    template <class V>
    auto function (const V& x) const
    {
        return (x(2) + 2) * x(1) * x(0) * x(0);
    }

    template <class V>
    auto ineqs (const V& x) const
    {
        using Scalar = impl::Scalar<V>;

        VecX<Scalar> g(4);

        g(0) = 1.0 - (std::pow(x(1), 3) * x(2)) / (7.1785 * std::pow(x(0), 4));
        g(1) = (4 * std::pow(x(1), 2) - x(0) * x(1)) / (12566 * x(1) * std::pow(x(0), 3) - std::pow(x(0), 4)) + 1.0 / (5.108 * std::pow(x(0), 2)) - 1.0;
        g(2) = 1.0 - (140.45 * x(0)) / (std::pow(x(1), 2) * x(2));
        g(3) = (x(1) + x(0)) / 1.5 - 1.0;

        return g;
    }

    template <class V>
    auto bounds () const
    {
        VecX<impl::Scalar<V>> lb(3), ub(3);

        lb << 0.05, 0.25, 2.0;
        ub << 2.0,  1.3, 15.0;

        return std::make_pair(lb, ub);
    }
};

} // namespace nlpp
