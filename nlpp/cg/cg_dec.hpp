#pragma once

#include "../helpers/helpers.hpp"
#include "../utils/optimizer.hpp"
#include "../line_search/strong_wolfe/strong_wolfe.hpp"
#include "projections_dec.hpp"


namespace nlpp
{

namespace impl
{

template <class Base_>
struct CG : public Base_
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base, Base_);
    using CGType = typename Base::CGType;
    using Base::cg;
    using Base::v;

    template <class Functions, class Domain, class Constraints>
    auto optimize (const Functions& funcs, const Domain& domain, const Constraints&) const
    {
        return optimize(funcs, domain.x0, cg, lineSearch, stop, output);
    }

    template <class Function, class V>
    std::tuple<V, impl::Scalar<V>, V, Status> optimize (const Function&, const V&, CGType, LineSearch, Stop, Output) const;
};

} // namespace impl

template <class Impl>
struct CGBase : public LineSearchOptimizer<Impl>
{
    using CGType = typename traits::Optimizer<Impl>::CGType;
    CGType cg;
    types::Float v = 0.1;     ///< The minimum factor of orthogonality that the current direction must have
};

template <class CGType = FR_PR, class LineSearch = ::nlpp::ls::StrongWolfe<>, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
struct CG : public impl::CG<CGBase<CG<CGType, LineSearch, Stop, Output>>>
{
};


namespace traits
{

using ::nlpp::wrap::Conditions;

template <class CGType_, class LineSearch_, class Stop_, class Output_>
struct Optimizer<CG<CGType_, LineSearch_, Stop_, Output_>>
{
    using CGType = CGType_;
    using LineSearch = LineSearch_;
    using Stop = Stop_;
    using Output = Output_;

    static constexpr Conditions conditions = Conditions::Function | Conditions::Gradient |
                                             Conditions::Start;
};

} // namespace traits

} // namespace nlpp