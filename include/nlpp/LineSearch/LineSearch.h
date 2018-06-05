/** @file
 * 	@brief Line search base class for CRTP
*/

#pragma once

#include "../Helpers/Helpers.h"

#include "../Helpers/Gradient.h"

#include "../Helpers/FiniteDifference.h"


namespace nlpp
{

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
template <class Impl, bool CalcNorm = false>
class LineSearch
{
public:

	
	template <class Function, class Vec>
	double impl (Function f, const Eigen::MatrixBase<Vec>& x, const Eigen::MatrixBase<Vec>& dir)
	{
		return static_cast<Impl&>(*this).lineSearch([&](double a)
		{
			static Vec gx(x.rows(), x.cols());

			auto fx = f(x + a * dir, static_cast<Vec&>(gx));

			return std::make_pair(fx, gx.dot(dir));
		});
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


private:

	LineSearch () {}

	friend Impl;

	int N;

	double xNorm;
};

} // namespace nlpp