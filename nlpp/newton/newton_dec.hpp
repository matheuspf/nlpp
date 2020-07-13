#pragma once

#include "../utils/optimizer.hpp"
#include "../line_search/strong_wolfe/strong_wolfe.hpp"
#include "factorizations_dec.hpp"


namespace nlpp
{

namespace impl
{

template <class Base_>
struct Newton : public Base_
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base, Base_);
    using Factorization = typename Base::Factorization;
    using Base::factorization;

    template <class Functions, class Domain, class Constraints>
    typename Domain::V optimize (const Functions& funcs, const Domain& domain, const Constraints&) const
    {
        return optimize(funcs, domain.x0, factorization, lineSearch, stop, output);
    }

    template <class Function, class V>
    V optimize (const Function&, V, Factorization, LineSearch, Stop, Output) const;
};

} // namespace impl


template <class Impl>
struct NewtonBase : public impl::LineSearchOptimizer<Impl>
{
    using Factorization = typename traits::Optimizer<Impl>::Factorization;
    Factorization factorization;
};

template <class Factorization = fact::SmallIdentity<>, class LineSearch = ls::StrongWolfe<>,
          class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
struct Newton : public impl::Newton<NewtonBase<Newton<Factorization, LineSearch, Stop, Output>>>
{
};

namespace traits
{

using ::nlpp::wrap::Conditions;

template <class Factorization_, class LineSearch_, class Stop_, class Output_>
struct Optimizer<::nlpp::Newton<Factorization_, LineSearch_, Stop_, Output_>>
{
    using Factorization = Factorization_;
    using LineSearch = LineSearch_;
    using Stop = Stop_;
    using Output = Output_;

    static constexpr Conditions conditions = Conditions::Function | Conditions::Gradient | Conditions::Hessian |
                                             Conditions::Start;
};

} // namespace traits


} // namespace nlpp
