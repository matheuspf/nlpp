/** @file
 * 	@brief Line search base class for CRTP
*/

#pragma once

#include "../Helpers/Helpers.h"

#include "../Helpers/Wrappers.h"

#include "../Helpers/FiniteDifference.h"

#include "../Helpers/Optimizer.h"


namespace nlpp
{

namespace wrap
{

template <class FunctionGradient, class V>
struct LineSearch
{
	using Float = typename V::Scalar;

	LineSearch (const FunctionGradient& f, const V& x, const V& d) : f(f), x(x), d(d), gx(x.rows(), x.cols())
	{
	}

	std::pair<Float, Float> operator () (Float a)
	{
		auto fx = f(x + a * d, gx);

		return std::make_pair(fx, gx.dot(d));
	}

	Float function (Float a)
	{
		return f.function(x + a * d);
	}

	Float gradient (Float a)
	{
		f.gradient(x + a * d, gx);

		return gx.dot(d);
	}


	FunctionGradient f;

	const V& x;
	const V& d;

	typename V::PlainObject gx;
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
template <class Impl, bool CalcNorm = true>
struct LineSearchBase
{
	template <class Function, class Vec>
	double impl (Function f, const Eigen::MatrixBase<Vec>& x, const Eigen::MatrixBase<Vec>& dir)
	{
		return static_cast<Impl&>(*this).lineSearch(wrap::LineSearch<Function, Vec>(f, x, dir));
	}


	template <class FunctionGradient, class Vec, std::enable_if_t<(wrap::IsFunctionGradient<FunctionGradient, Vec>::value >= 0), int> = 0>
	double operator () (const FunctionGradient& f, const Eigen::MatrixBase<Vec>& x, const Eigen::MatrixBase<Vec>& dir)
	{
		return impl(f, x, dir);
	}

	template <class Function, class Gradient, class Vec>
	double operator () (Function f, Gradient g, const Eigen::MatrixBase<Vec>& x, const Eigen::MatrixBase<Vec>& dir)
	{
		return impl(wrap::functionGradient(f, g), x, dir);
	}

	template <class Function, class Vec, std::enable_if_t<(wrap::IsFunction<Function, Vec>::value >= 0 && wrap::IsFunctionGradient<Function, Vec>::value < 0), int> = 0>
	double operator () (Function f, const Eigen::MatrixBase<Vec>& x, const Eigen::MatrixBase<Vec>& dir)
	{
		return operator()(f, fd::gradient(f), x, dir);
	}


	template <class Function, class Gradient>
	double operator () (Function f, Gradient g, double x, double dir = 1.0)
	{
		return operator()([&](double a){ return f(x + a * dir); },
						  [&](double a){ return g(x + a * dir) * dir; });
	}

	template <class Function>
	double operator () (Function f, double x, double dir = 1.0)
	{
		return operator()(f, fd::gradient(f), x, dir);
	}


//private:

	LineSearchBase () {}

	friend Impl;
};




template <class Impl>
struct LineSearch : public LineSearchBase<LineSearch<Impl>>
{
	template <class Function>
	double lineSearch (Function f)
	{
		return static_cast<Impl&>(*this).lineSearch(f);
	}
};


namespace poly
{

template <class Function>
struct LineSearch : public LineSearchBase<LineSearch<Function>>
{
	virtual ~LineSearch () {}

	virtual double lineSearch (Function f) = 0;

	virtual LineSearch* clone () const = 0;
};




} // namespace poly



template <class Impl, bool Polymorphic, typename... Args>
using LineSearchPick = std::conditional_t<Polymorphic, poly::LineSearch<Args...>, LineSearch<Impl>>;



} // namespace nlpp
