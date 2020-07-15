#pragma once

#include "../../utils/optimizer.hpp"
#include "../../line_search/strong_wolfe/strong_wolfe.hpp"
#include "../bfgs/initial_hessian_dec.hpp"


namespace nlpp
{

namespace impl
{

template <class Base_>
struct LBFGS : public Base_
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base, Base_);
    using InitialHessian = typename Base::InitialHessian;
    using Base::initialHessian, Base::m;

    template <class Functions, class Domain, class Constraints>
    typename Domain::V optimize (const Functions& funcs, const Domain& domain, const Constraints&) const
    {
        return optimize(funcs, domain.x0, initialHessian, lineSearch, stop, output);
    }

    template <class Function, class V>
    V optimize (const Function&, V, InitialHessian, LineSearch, Stop, Output) const;

    template <class Function, class V, class U>
    V direction (const Function&, const V&, const V&, const Eigen::MatrixBase<U>&, const std::deque<V>&, const std::deque<V>&) const;
};

} // namespace impl


template <class Impl>
struct LBFGSBase : public impl::LineSearchOptimizer<Impl>
{
    using InitialHessian = typename traits::Optimizer<Impl>::InitialHessian;
    InitialHessian initialHessian;
    int m;
};

template <class InitialHessian = BFGSDiagonal<>, class LineSearch = ls::StrongWolfe<>,
          class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
struct LBFGS : public impl::LBFGS<LBFGSBase<LBFGS<InitialHessian, LineSearch, Stop, Output>>>
{
};

namespace traits
{

using ::nlpp::wrap::Conditions;

template <class InitialHessian_, class LineSearch_, class Stop_, class Output_>
struct Optimizer<::nlpp::LBFGS<InitialHessian_, LineSearch_, Stop_, Output_>>
{
    using InitialHessian = InitialHessian_;
    using LineSearch = LineSearch_;
    using Stop = Stop_;
    using Output = Output_;

    static constexpr Conditions conditions = Conditions::Function | Conditions::Gradient |
                                             Conditions::Start;
};

} // namespace traits

} // namespace nlpp