#pragma once

#include "utils/optimizer.hpp"
#include "LineSearch/Goldstein/Goldstein.hpp"


namespace nlpp
{

namespace impl
{

template <class Base_>
struct GradientDescent : public Base_
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base, Base_);

    template <class F, class V>
    V optimize (const F&, V);
};

} // namespace impl

template <class LineSearch = Goldstein<>, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
struct GradientDescent : public impl::GradientDescent<LineSearchOptimizer<GradientDescent<LineSearch, Stop, Output>>>
{
};


namespace traits
{

template <class LineSearch_, class Stop_, class Output_>
struct LineSearchOptimizer<GradientDescent<LineSearch_, Stop_, Output_>>
{
    using LineSearch = LineSearch_;
    using Stop = Stop_;
    using Output = Output_;
};

} // namespace traits

} // namespace nlpp
