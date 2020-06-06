#pragma once

#include "helpers/helpers.hpp"


namespace nlpp
{

template <typename Float = types::Float>
struct FirstOrderStep
{
    FirstOrderStep (Float a0 = 1.0, Float aMin = std::sqrt(constants::eps_<Float>)) : a0(a0), aMin(aMin), initialized(false) {}

    Float operator () ()
    {
        return a0;
    }

    template <class LS>
    Float operator () (const LS& ls, Float f1, Float g1)
    {
        Float a = a0;

        if(initialized)
        {
            a = (2 * (f1 - f0)) / g0;
            a = std::min(a0, 1.01 * a);
            a = std::max(a, aMin);
        }

        if(std::isnan(a) || std::isinf(a))
            a = a0;

        initialized = true;
        std::tie(f0, g0) = std::tie(f1, g1);

        return a;
    }

    Float a0;
    Float aMin;
    Float f0;
    Float g0;
    bool initialized = false;
};

} // namespace nlpp