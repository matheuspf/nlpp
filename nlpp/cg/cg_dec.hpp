#pragma once

#include "../helpers/helpers.hpp"
#include "../utils/optimizer.hpp"
#include "../LineSearch/StrongWolfe/StrongWolfe.hpp"
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
    V optimize (const Function&, V, LineSearch, Stop, Output) const;
};

} // namespace impl

template <class Impl>
struct CGBase : public LineSearchOptimizer<Impl>
{
    typename traits::Optimizer<Impl>::CGType cg;
    double v = 0.1;     ///< The minimum factor of orthogonality that the current direction must have
};

template <class CGType = FR_PR, class LineSearch = StrongWolfe<>, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
struct CG : public impl::CG<CGBase<CG<CGType, LineSearch, Stop, Output>>>
{
};

// template <class CGType = FR_PR, class LineSearch = StrongWolfe<>, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
// using CG = impl::CG<CGBase<CG<CGType, LineSearch, Stop, Output>, CGType, LineSearch, Stop, Output>>;


namespace traits
{

template <class CGType_, class LineSearch_, class Stop_, class Output_>
struct Optimizer<CG<CGType_, LineSearch_, Stop_, Output_>>
{
    using CGType = CGType_;
    using LineSearch = LineSearch_;
    using Stop = Stop_;
    using Output = Output_;
};

} // namespace traits

} // namespace nlpp