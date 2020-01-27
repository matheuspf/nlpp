#pragma once

#include "../utils/optimizer.hpp"
#include "../LineSearch/StrongWolfe/StrongWolfe.hpp"
#include "factorizations_dec.hpp"


namespace nlpp
{

namespace impl
{

template <class Base_>
struct Newton : public Base_
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base, Base_);
    using Base::factorization;

	template <class Function, class Hessian, class V>
	V optimize (const Function& f, const Hessian& hess, V x);
};

} // namespace impl


template <class Impl>
struct NewtonBase : public impl::LineSearchOptimizer<Impl>
{
    typename traits::Optimizer<Impl>::Factorization factorization;
};

template <class Factorization = fact::SmallIdentity<>, class LineSearch = StrongWolfe<>,
		class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
struct Newton : public impl::Newton<NewtonBase<Newton<Factorization, LineSearch, Stop, Output>>>
{
};

namespace traits
{

template <class Factorization_, class LineSearch_, class Stop_, class Output_>
struct Optimizer<::nlpp::Newton<Factorization_, LineSearch_, Stop_, Output_>>
{
    using Factorization = Factorization_;
    using LineSearch = LineSearch_;
    using Stop = Stop_;
    using Output = Output_;
};

} // namespace traits


} // namespace nlpp
