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
struct GradientDescent : public ::nlpp::params::GradientOptimizer<LineSearch, Stop, Output>
{
	CPPOPT_USING_PARAMS(Params, ::nlpp::params::GradientOptimizer<LineSearch, Stop, Output>);
	using Params::Params;
};

} // namespace params


template <class LineSearch = ::nlpp::Goldstein<>, class Stop = ::nlpp::stop::GradientOptimizer<>, class Output = ::nlpp::out::GradientOptimizer<0>>
struct GradientDescent : public params::GradientDescent<LineSearch, Stop, Output>
{
	CPPOPT_USING_PARAMS(Params, params::GradientDescent<LineSearch, Stop, Output>);
	using Params::Params;
	

	template <class Function, class V>
	V optimize (Function f, V x)
	{
		impl::Scalar<V> fx;
		V gx, dir;
		
		std::tie(fx, gx) = f(x);

		stop.init(*this, x, fx, gx);
		output.init(*this, x, fx, gx);

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

		output.finish(*this, x, fx, gx);

		return x;
	}
};

} // namespace impl


template <class LineSearch = Goldstein<>, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<0>>
struct GradientDescent : public impl::GradientDescent<LineSearch, Stop, Output>,
						 public GradientOptimizer<GradientDescent<LineSearch, Stop, Output>>
{
	CPPOPT_USING_PARAMS(Impl, impl::GradientDescent<LineSearch, Stop, Output>);
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
struct GradientDescent : public ::nlpp::impl::GradientDescent<::nlpp::poly::LineSearch_<>, ::nlpp::stop::poly::GradientOptimizer_<>, ::nlpp::out::poly::GradientOptimizer_<>>,
						 public ::nlpp::poly::GradientOptimizer<V>
{
	CPPOPT_USING_PARAMS(Impl, ::nlpp::impl::GradientDescent<::nlpp::poly::LineSearch_<>, ::nlpp::stop::poly::GradientOptimizer_<>, ::nlpp::out::poly::GradientOptimizer_<>>);
	using Impl::Impl;

	virtual V optimize (::nlpp::wrap::poly::FunctionGradient<V> f, V x)
	{
		return Impl::optimize(f, x);
	}

	virtual GradientDescent* clone_impl () const { return new GradientDescent(*this); }
};

} // namespace poly


} // namespace nlpp
