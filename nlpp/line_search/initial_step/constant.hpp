#pragma once

#include "helpers/helpers.hpp"

// #include "InitialStep.h"


namespace nlpp
{

template <typename Float = types::Float>
struct ConstantStep
{
    ConstantStep (Float a0 = 1.0) : a0(a0) {}

    Float operator () ()
    {
        return a0;
    }

    template <class LS, class... Args>
    Float operator () (const LS&, const Args&...)
    {
        return a0;
    }

    Float a0;
};

} // namespace nlpp