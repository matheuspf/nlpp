/** @file
 *  @brief Nonlinear Conjugate Gradient
 * 
 *  @details An implementation of the nonlinear CG algorithm, described in chapter 5 of NOCEDAL.
 * 
 * 	The nonlinear version of the CG algorithms does not make any assumption about the objective function 
 * 	(it only needs to be smooth). Also, it has a quite concise structure, only changing the way to calculate
 *  the scalar factor of the previous directions that should be added to the current gradient.
 * 
 *  The methods implemented are: Fletcher-Reeves (FR), Polak-Ribière and variants (PR, PR_abs, PR_Plus, FR_PR),
 * 	Hestenes-Stiefel (HS)
*/

#pragma once
#include "../Helpers/Helpers.h"
#include "../Helpers/Optimizer.h"
#include "../LineSearch/StrongWolfe/StrongWolfe.h"


namespace impl
{

template <class CGType, class Base_>
struct CG : public Base_
{
	CPPOPT_USING_PARAMS(Base, Base_);
	using Base::Base;


	template <class Function, class V>
	V optimize (Function f, V x)
	{
		V fa, dir, fb(x.rows(), x.cols());

		impl::Scalar<V> fxOld, fx;

		std::tie(fxOld, fa) = f(x);

		dir = -fa;

		for(int iter = 0; iter < stop.maxIterations(); ++iter)
		{
			double alpha = lineSearch(f, x, dir);
			
			x = x + alpha * dir;

			fx = f(x, fb);
			

			if(stop(*this, x, fx, fb))
				break;

			if((fa.dot(fb) / fb.dot(fb)) >= v)
				dir = -fb;

			else
				dir = -fb + cg(fa, fb, dir) * dir;


			fxOld = fx;

			fa = fb;

			output(*this, x, fx, fb);
		}

		return x;
	}


	CGType cg;			///< The functor that calculates the search direction, given the current gradient and the previous directions

	double v = 0.1;		///< The minimum factor of orthogonality that the current direction must have
};

} // namespace impl


template <class CGType = FR_PR, class LineSearch = StrongWolfe<>, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
using CG = impl::CG<CGType, LineSearchOptimizer<impl::CG<CGType, LineSearch, Stop, Output>, LineSearch, Stop, Output>>;


namespace poly
{

template <class CGType = ::nlpp::FR_PR, class V = ::nlpp::Vec>
struct CG : public ::nlpp::impl::CG<CGType, ::nlpp::poly::LineSearchOptimizer<V>>
{
	using Impl = ::nlpp::impl::CG<CGType, ::nlpp::poly::LineSearchOptimizer<V>>;
	using Impl::Impl;

	virtual V optimize (::nlpp::wrap::poly::FunctionGradient<V> f, V x)
	{
		return Impl::optimize(f, x);
	}

	virtual CG* clone_impl () const { return new CG(*this); }
};

} // namespace poly

} // namespace nlpp