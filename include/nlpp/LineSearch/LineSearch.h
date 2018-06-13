/** @file
 * 	@brief Line search base class for CRTP
*/

#pragma once

#include "../Helpers/Helpers.h"

#include "../Helpers/Gradient.h"

#include "../Helpers/FiniteDifference.h"

#include "../Helpers/Optimizer.h"


namespace nlpp
{

namespace wrap
{

template <class FunctionGradient, class Vec>
struct LineSearch
{
	LineSearch (FunctionGradient f, const Vec& x, const Vec& d) : f(f), x(x), d(d) {}

	auto operator () (double a)
	{
		static Vec gx(x.rows(), x.cols());

		auto fx = f(x + a * d, static_cast<Vec&>(gx));

		return std::make_pair(fx, gx.dot(d));
	}

	FunctionGradient f;

	Vec x;
	Vec d;
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
		wrap::LineSearch<Function, Vec> ls(f, x, dir);

		return static_cast<Impl&>(*this).lineSearch(ls);
	}


	/** @name
	 * 	@brief Decide whether to calculate the norm of @c x or not
	*/
	//@{
	template <class Function, class Vec, bool CalcNorm_ = CalcNorm, std::enable_if_t<!CalcNorm_, int> = 0>
	double delegate (Function f, const Eigen::MatrixBase<Vec>& x, const Eigen::MatrixBase<Vec>& dir)
	{
		N = x.size();

		return impl(f, x, dir);
	}

	template <class Function, class Vec, bool CalcNorm_ = CalcNorm, std::enable_if_t<CalcNorm_, int> = 0>
	double delegate (Function f, const Eigen::MatrixBase<Vec>& x, const Eigen::MatrixBase<Vec>& dir)
	{
		xNorm = x.norm();

		return delegate<Function, Vec, false>(f, x, dir);
	}
	//@}


	template <class FunctionGradient, class Vec, std::enable_if_t<wrap::IsFunctionGradient<FunctionGradient, Vec>::value, int> = 0>
	double operator () (const FunctionGradient& f, const Eigen::DenseBase<Vec>& x, const Eigen::DenseBase<Vec>& dir)
	{
		return delegate(wrap::functionGradient(f), x.eval(), dir.eval());
	}

	template <class Function, class Gradient, class Vec>
	double operator () (Function f, Gradient g, const Eigen::DenseBase<Vec>& x, const Eigen::DenseBase<Vec>& dir)
	{
		return delegate(wrap::functionGradient(f, g), x.eval(), dir.eval());
	}

	template <class Function, class Vec, std::enable_if_t<wrap::IsFunction<Function, Vec>::value, int> = 0>
	double operator () (Function f, const Eigen::DenseBase<Vec>& x, const Eigen::DenseBase<Vec>& dir)
	{
		return operator()(f, fd::gradient(f), x, dir);
	}


	template <class Function, class Gradient>
	double operator () (Function f, Gradient g, double x, double dir = 1.0)
	{
		N = 1;
		xNorm = std::abs(x);

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

	int N;

	double xNorm;
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
