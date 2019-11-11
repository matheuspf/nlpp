#pragma once

#include "../Helpers/Helpers.h"
#include "projections_dec.hpp"


namespace nlpp
{

namespace impl
{

template <class Base_>
struct CG : public Base_
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base, Base_);
    using CG::cg;

    template <class Function, class V>
    V optimize (Function f, V x);
};

} // namespace impl

template <class Impl, class CGType, class LineSearch, class Stop, class Output>
struct CGBase : public LineSearchOptimizer<Impl, LineSearch, Stop, Output>
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base, Base_);
    CGType cg;
};



template <class CGType = FR_PR, class LineSearch = StrongWolfe<>, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>


using CG = impl::CG<CGType, LineSearchOptimizer<impl::CG<CGType, LineSearch, Stop, Output>, LineSearch, Stop, Output>>;


} // namespace nlpp