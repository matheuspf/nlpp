#pragma once

#include "../helpers/helpers.hpp"

#include "../Helpers/Optimizer.h"

#include "../LineSearch/StrongWolfe/StrongWolfe.h"

#include "Factorizations.h"



namespace nlpp
{

namespace impl
{

template <class Factorization, class Base_>
struct Newton : public Base_
{
	CPPOPT_USING_PARAMS(Base, Base_);
	using Base::Base;

	template <class Function, class Hessian, class V>
	V optimize (Function f, Hessian hess, V x)
	{
		impl::Scalar<V> fx;
		V gx;

		std::tie(fx, gx) = f(x);

		for(int iter = 0; iter < stop.maxIterations(); ++iter)
		{
			auto dir = factorization(gx, hess(x));

			auto alpha = lineSearch(f, x, dir);

			x = x + alpha * dir;

			fx = f(x, gx);


			if(stop(*this, x, fx, gx))
				break;

			output(*this, x, fx, gx);
		}

		return x;
	}

	Factorization factorization;
};

} // namespace impl


template <class Factorization = fact::SmallIdentity<>, class LineSearch = StrongWolfe<>,
		class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
struct Newton : public impl::Newton<Factorization, LineSearchOptimizer<Newton<Factorization, LineSearch, Stop, Output>, LineSearch, Stop, Output>>
{
	using Impl = impl::Newton<Factorization, LineSearchOptimizer<Newton<Factorization, LineSearch, Stop, Output>, LineSearch, Stop, Output>>;
	using Impl::Impl;

	template <class Function, class Hessian, class V>
	V optimize (Function f, Hessian hess, V x)
	{
		return Impl::optimize(f, hess, x);
	} 

	template <class Function, class V>
	V optimize (Function f, V x)
	{
		return optimize(f, ::nlpp::fd::hessian(f), x);
	}
};


// namespace poly
// {

// template <class Factorization = ::nlpp::fact::SmallIdentity<>, class V = ::nlpp::Vec>
// struct Newton : public impl::Newton<::nlpp::poly::GradientOptimizer<V, impl::params::Newton<::nlpp::params::poly::LineSearchOptimizer_, Factorization>>>
// {
// 	CPPOPT_USING_PARAMS(Impl, impl::Newton<::nlpp::poly::GradientOptimizer<V, impl::params::Newton<::nlpp::params::poly::LineSearchOptimizer_, Factorization>>>)
// 	using Impl::Impl;

// 	virtual V optimize (::nlpp::wrap::poly::FunctionGradient<V> f, ::nlpp::wrap::poly::Hessian<V> hess, V x)
// 	{
// 		return Impl::optimize(f, hess, x);
// 	}

// 	virtual V optimize (::nlpp::wrap::poly::FunctionGradient<V> f, V x)
// 	{
// 		return optimize(f, ::nlpp::wrap::poly::Hessian<V>(::nlpp::fd::hessian(f.func)), x);
// 	}


// 	virtual Newton* clone_impl () const { return new Newton(*this); }
// };

// } // namespace poly


} // namespace nlpp
