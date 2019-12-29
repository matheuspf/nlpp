/** @file
 * 	@brief Line search base class for CRTP
*/

#pragma once

#include "helpers/helpers.hpp"
#include "utils/wrappers.hpp"
#include "utils/finiteDifference.hpp"
#include "InitialStep/Constant.h"



namespace nlpp
{

namespace wrap
{

/// A utility to wrap a vector function to a scalar function along a direction @c d
template <class FunctionGradient, class V>
struct LineSearch
{
	using Float = ::nlpp::impl::Scalar<V>;

	LineSearch (const FunctionGradient& f, const V& x, const V& d) : f(f), x(x), d(d)
	{
	}

	std::pair<Float, Float> operator () (Float a)
	{
		V xn = x + a * d;

		auto fx = f.function(xn);
		//auto gx = f.directional(xn, d, fx);
		auto gx = f.gradient(xn, d);

		return std::make_pair(fx, gx);
	}

	Float function (Float a)
	{
		return f.function(x + a * d);
	}

	Float gradient (Float a)
	{
		return f.gradient(x, d);
	}

	FunctionGradient f;
	const V& x;
	const V& d;
};

} // namespace wrap



/** @brief Line search base class for CRTP
 *  
 *  @details Delegate the call to the base class @c Impl after projecting the given function and gradient into
 * 			 the direction dir. That is:
 * 
 * 			 - @f$ f'(a) = f(x + a * dir) @f$
 * 			 - @f$ g'(a) = g(x + a * dir) \intercall dir @f$
 * 
 * 			So we can now use f' and g' exactly as if they were unidimensional scalar functions. Also, wraps the gradient 
 * 			or function/gradient calls before projection.
 * 
 *  @tparam Impl The actual line search implementation
 * 	@tparam Whether we must save the norm of the given vector before delegating the calls
*/
template <class Impl>
struct LineSearch
{
	template <class V>
	static constexpr bool isImplPoly = std::is_same<Impl, poly::LineSearchBase<V>>::value || std::is_same<Impl, poly::LineSearch_<V>>::value;

    void initialize () 
	{
		static_cast<Impl&>(*this).initialize();
	}

    template <class Function, class V>
	::nlpp::impl::Scalar<V> impl (Function f, const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& dir)
	{
		return static_cast<Impl&>(*this).lineSearch(wrap::LineSearch<Function, V>(f, x, dir));
	}


	template <class Function, class V>
	::nlpp::impl::Scalar<V> operator () (Function f, const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& dir)
	{
		if constexpr(isImplPoly<V>)
			return impl(wrap::poly::makeFuncGrad<V>(f), x, dir);

		else
	    	return impl(wrap::makeFuncGrad<V>(f), x, dir);
	}

	template <class Function, class Gradient, class V>
	::nlpp::impl::Scalar<V> operator () (Function f, Gradient g, const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& dir)
	{
		if constexpr(isImplPoly<V>)
			return impl(wrap::poly::makeFuncGrad<V>(f, g), x, dir);

		else
	    	return impl(wrap::makeFuncGrad(f, g), x, dir);
	}

    // template <class Function, class Gradient, typename Float, std::enable_if_t<std::is_floating_point<Float>::value, int> = 0>
	// auto operator () (Function f, Gradient g, Float x, Float dir = 1.0)
	// {
	// 	return operator()([&](Float a){ return f(x + a * dir); },
	// 					  [&](Float a){ return g(x + a * dir) * dir; });
	// }

	// template <class Function, typename Float, typename Float, std::enable_if_t<std::is_floating_point<Float>::value, int> = 0>
	// auto operator () (Function f, Float x, Float dir = 1.0)
	// {
	// 	return operator()(f, fd::gradient(f), x, dir);
	// }
};



namespace poly
{

template <class V = ::nlpp::Vec>
struct LineSearchBase : public ::nlpp::poly::CloneBase<LineSearchBase<V>>
{
	virtual ~LineSearchBase ()	{}

	virtual void initialize () = 0;

	virtual ::nlpp::impl::Scalar<V> lineSearch (::nlpp::wrap::LineSearch<::nlpp::wrap::FunctionGradient<::nlpp::wrap::impl::FunctionGradient<V>>, V>) = 0;
};


template <class V = ::nlpp::Vec>
struct LineSearch_ : public ::nlpp::poly::PolyClass<LineSearchBase<V>>,
					 public ::nlpp::LineSearch<LineSearch_<V>>
{
	NLPP_USING_POLY_CLASS(LineSearch_, Base, ::nlpp::poly::PolyClass<LineSearchBase<V>>);

	LineSearch_ () : Base(std::make_unique<StrongWolfe<V, ConstantStep<::nlpp::impl::Scalar<V>>>>()) {}

    void initialize ()
	{
		return impl->initialize();
	}

	::nlpp::impl::Scalar<V> lineSearch (::nlpp::wrap::LineSearch<::nlpp::wrap::FunctionGradient<::nlpp::wrap::impl::FunctionGradient<V>>, V> f)
	{
		return impl->lineSearch(f);
	}
};

} // namespace poly



template <typename Float = types::Float, class InitialStep = ConstantStep<Float>>
struct LineSearchBase
{
	LineSearchBase (const InitialStep& initialStep) : initialStep(initialStep) {}


    void initialize ()
	{
		initialStep.initialize();
	}


	Float aStart;
	Float f0;
	Float g0;

	InitialStep initialStep;
};



} // namespace nlpp
