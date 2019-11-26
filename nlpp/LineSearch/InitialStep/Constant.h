#pragma once

#include "helpers/helpers.hpp"

// #include "InitialStep.h"


namespace nlpp
{

template <typename Float>
struct ConstantStep
{
    ConstantStep (Float a0 = 1.0) : a0(a0) {}

    void initialize () {}

    Float operator () (...) const { return a0; }

    Float a0;	
};

} // namespace nlpp