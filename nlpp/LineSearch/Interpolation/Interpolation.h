#pragma once

#include "helpers/helpers.hpp"


namespace nlpp
{

template <typename Float>
Float interpolate (Float a, Float b, Float factor = 0.5)
{
    return factor * a + (1.0 - factor) * b;
}

template <typename Float>
Float interpolate (Float a, Float b, Float fa, Float fb, Float ga)
{
    Eigen::Matrix<Float, 3, 3> A;
    Eigen::Matrix<Float, 3, 1> c;

    A << 2*a, 1, 0,
         a*a, a, 1,
         b*b, b, 1;

    c << ga, fa, fb;

    Eigen::Matrix<Float, 3, 1> x = -A.inverse() * c;
    
    if(std::any_of(x.data(), x.data() + x.size(), [](Float xi){ return std::isnan(xi) || std::isinf(xi); }))
        return interpolate(a, b);

    Float res = -x(1) / (2 * x(0) + constants::eps_<Float>);

    if(std::isnan(res) || std::isinf(res) || res < std::min(a, b) + constants::eps_<Float> || res > std::max(a, b) - constants::eps_<Float>)
        return interpolate(a, b);

    // handy::print("\n\nA: ", res, "\n\n");

    return res;
}

template <typename Float>
Float interpolate (Float a, Float b, Float fa, Float fb, Float ga, Float gb)
{
    Eigen::Matrix<Float, 4, 4> A;
    Eigen::Matrix<Float, 4, 1> c;

    A << 3*a*a, 2*a, 1, 0,
         3*b*b, 2*b, 1, 0,
         a*a*a, a*a, a, 1,
         b*b*b, b*b, b, 1;

    c << ga, gb, fa, fb;

    Eigen::Matrix<Float, 4, 1> x = -A.inverse() * c;

    // handy::print("HERE");
    
    if(std::any_of(x.data(), x.data() + x.size(), [](Float xi){ return std::isnan(xi) || std::isinf(xi); }))
    {
        // handy::print("\n\nLOL\n\n");
        return interpolate(a, b);
    }

    Float delta = x(1) * x(1) - 3 * x(0) * x(2);

    if(delta < 0.0)
    {
        // handy::print("\n\nLAL\n\n");
        return interpolate(a, b);
    }

    Float res0 = (-x(1) - std::sqrt(delta)) / (3 * x(0));
    Float res1 = (-x(1) + std::sqrt(delta)) / (3 * x(0));

    auto isFeasible = [&](Float res)
    {
        return !std::isnan(res) && !std::isinf(res) && 
               res > std::min(a, b) + 0.1 * std::abs(b - a) &&
               res < std::max(a, b) - 0.1 * std::abs(b - a);
    };

    if(isFeasible(res0))
    {
        // handy::print("\n\nB: ", res0, "\n\n");
        return res0;
    }

    if(isFeasible(res1))
    {
        // handy::print("\n\nC: ", res1, "\n\n");
        return res1;
    }

    
    // handy::print("\n\nLAL\n\n");

   return interpolate(a, b);
}


} // namespace nlpp