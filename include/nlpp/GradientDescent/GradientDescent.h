#pragma once

#include "../Helpers/Helpers.h"

#include "../Helpers/Optimizer.h"

#include "../LineSearch/Goldstein/Goldstein.h"

#include "../LineSearch/StrongWolfe/StrongWolfe.h"


namespace nlpp
{

namespace impl
{

namespace params
{

template <class LineSearch = ::nlpp::Goldstein<>, class Stop = ::nlpp::stop::GradientOptimizer<>, class Output = ::nlpp::out::GradientOptimizer<0>>
struct GradientDescent : public ::nlpp::params::LineSearchOptimizer<LineSearch, Stop, Output>
{
	CPPOPT_USING_PARAMS(Params, ::nlpp::params::LineSearchOptimizer<LineSearch, Stop, Output>);
	using Params::Params;
};

} // namespace params


template <class Params_>
struct GradientDescent : public Params_
{
	CPPOPT_USING_PARAMS(Params, Params_);
	using Params::Params;
	

	template <class Function, class V>
	V optimize (Function f, V x)
	{
		impl::Scalar<V> fx;
		V gx, dir;
		
		std::tie(fx, gx) = f(x);

		for(int iter = 0; iter < stop.maxIterations(); ++iter)
		{
			dir = -gx;

			auto alpha = lineSearch(f, x, dir);

			x = x + alpha * dir;

			fx = f(x, gx);

			if(stop(*this, x, fx, gx))
				break;

			output(*this, x, fx, gx);
		}

		return x;
	}
};

} // namespace impl


template <class LineSearch = StrongWolfe<>, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
struct GradientDescent : public impl::GradientDescent<params::LineSearchOptimizer<LineSearch, Stop, Output>>,
						 public GradientOptimizer<GradientDescent<LineSearch, Stop, Output>>
{
	CPPOPT_USING_PARAMS(Impl, impl::GradientDescent<params::LineSearchOptimizer<LineSearch, Stop, Output>>);
	using Impl::Impl;

	template <class Function, class V>
	V optimize (Function f, V x)
	{
		return Impl::optimize(f, x);
	}
};


namespace poly
{

template <class V = ::nlpp::Vec>
struct GradientDescent : public ::nlpp::impl::GradientDescent<::nlpp::poly::GradientOptimizer<V>>
{
	CPPOPT_USING_PARAMS(Impl, ::nlpp::impl::GradientDescent<::nlpp::poly::GradientOptimizer<V>>);
	using Impl::Impl;
	using Vec = V;

	virtual V optimize (::nlpp::wrap::poly::FunctionGradient<V> f, V x)
	{
		return Impl::optimize(f, x);
	}

	virtual GradientDescent* clone_impl () const { return new GradientDescent(*this); }
};

} // namespace poly


} // namespace nlpp
