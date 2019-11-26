#pragma once

#include "../helpers/helpers.hpp"
#include "../utils/optimizer.hpp"
#include "../LineSearch/StrongWolfe/StrongWolfe.h"
#include "projections_dec.hpp"


namespace nlpp
{

namespace impl
{

template <class Base_>
struct CG : public Base_
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base, Base_);
    using Base::cg;
    using Base::v;

    template <class Function, class V>
    V optimize (Function, V);
};

} // namespace impl

template <class Impl, class CGType, class LineSearch, class Stop, class Output>
struct CGBase : public LineSearchOptimizer<Impl, LineSearch, Stop, Output>
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base, LineSearchOptimizer<Impl, LineSearch, Stop, Output>);

    CGType cg;
    double v = 0.1;     ///< The minimum factor of orthogonality that the current direction must have
};

template <class CGType = FR_PR, class LineSearch = StrongWolfe<>, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
struct CG : public impl::CG<CGBase<CG<CGType, LineSearch, Stop, Output>, CGType, LineSearch, Stop, Output>>
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base, impl::CG<CGBase<CG<CGType, LineSearch, Stop, Output>, CGType, LineSearch, Stop, Output>>);
    using Base::optimize;
};

// template <class CGType = FR_PR, class LineSearch = StrongWolfe<>, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
// using CG = impl::CG<CGBase<CG<CGType, LineSearch, Stop, Output>, CGType, LineSearch, Stop, Output>>;


} // namespace nlpp