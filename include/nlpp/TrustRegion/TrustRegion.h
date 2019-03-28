#pragma once

#include "../Helpers/Helpers.h"
#include "../Helpers/FiniteDifference.h"
#include "../Helpers/Parameters.h"



namespace nlpp
{

namespace params
{

template <class LocalOptimizer, class Params, typename Float = types::Float>
struct TrustRegion : public Params
{
	using Params::Params;
	using Params::stop;
	using Params::output;

	TrustRegion (Float delta0 = 10.0, Float alpha = 0.25, Float beta = 2.0, Float eta = 0.1, Float maxDelta = 1e2) :
				 delta0(delta0), alpha(alpha), beta(beta), eta(eta), maxDelta(maxDelta) {}


	LocalOptimizer localOptimizer;

	Float delta0;
	Float alpha;
	Float beta;
	Float eta;
	Float maxDelta;
};

} // namespace params


namespace impl
{

template <class LocalOptimizer, class Params_, typename Float = types::Float>
struct TrustRegion : public ::nlpp::params::TrustRegion<Params_>
{
	using Params = ::nlpp::params::TrustRegion<Params_>;
	using Params::Params;


	template <class Function, class Hessian, class V>
	V optimize (Function function, Hessian hessian, V x)
	{
		double delta = delta0;

		double fx = function(x);
		Vec gx = gradient(x);
		Mat hx = hessian(x);



		for(int iter = 0; iter < static_cast<Impl&>(*this).stop.maxIterations(); ++iter)
		{
			/** Making a call to the actual function that generates the direction whitin the trust region.
			  *	I am using CRTP here, so the 'Impl' class inherits from this class. **/
			Vec dir = static_cast<Impl&>(*this).direction(function, gradient, hessian, x, delta, fx, gx, hx);

			Vec y = x + dir;			/// New point and
			double fy = function(y);	/// and its fitness

			double aRed = (fx - fy);	/// Actual reduction 
			double pRed = -(gx.dot(dir) + 0.5 * dir.transpose() * hx * dir);	/// Predicted reduction

			/** Ratio between actual and predicted reduction. 'pRed' SHOULD always be positive, but I am also 
			 *  handling the case where the trust region direction fails miserably and returns a worst solution.
			 *  That 'abs' actually should not be there.
			**/
			double rho = aRed / std::max(constants::eps, std::abs(pRed));

			//db(delta, "       ", function(x), "      ", aRed, "      ", pRed, "       ", rho, "       ", x.transpose(), "         ", dir.transpose(), "\n\n\n");
			//db((fx - fy), "      ", (-gx.dot(dir) - 0.5 * dir.transpose() * hx * dir), "       ", x.transpose(), "     ", dir.transpose(), "\n\n\n");

			if(std::isnan(rho))
			{
				handy::print("FAIL HARD");
				handy::print("\n\n\n", fx, "     ", fy, "          ", gx.transpose(), "          ", dir, "\n\n");
				std::cout << hx << "\n\n"; exit(0);
			}



			/// If this is the maximum allowed value for delta and we had this small improvement, theres nothing else to do
			if(aRed < constants::eps && delta == maxDelta)
				return x;

			/// If rho is smaller than this threshold, reduce the trust region size
			if(rho < alpha)
				delta = alpha * delta;

			/** If rho is greater than this other threshold AND the direction is at
			  *	the border of the trust region, then increase the trust region size.
			  *
			  * TODO: change that aproximation (1e-4) and return a flag telling if the direction is at the border.
			**/
			else if(rho > 1.0 - alpha && std::abs(dir.norm() - delta) < 1e-4)
				delta = std::min(beta * delta, maxDelta);


			/// Too small trust region, go home. Nothing else to do
			if(delta < 2 * constants::eps) 
				return x;

			if(static_cast<Impl&>(*this).stop())

			/// We only update the actual x if rho is greater than eta and the function value is actually reduced
			if(rho > eta)
			{
				x = y;
				fx = fy;
				gx = gradient(x);
				hx = hessian(x);
			}
		}

		return x;
	}
};

} // namespace impl


template <class LocalMinimizer, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>, typename Float = types::Float>
struct TrustRegion : public impl::TrustRegion<LocalMinimizer, params::Optimizer<Stop, Output>, Float>,
					 public GradientOptimizer<TrustRegion<LocalMinimizer, Stop, Output, Float>>
{
	using Impl = impl::TrustRegion<LocalMinimizer, params::Optimizer<Stop, Output>, Float>;
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


namespace poly
{

template <class LocalMinimizer, class V = ::nlpp::Vec>
struct TrustRegion : public ::nlpp::impl::TrustRegion<LocalMinimizer, params::Optimizer_>
{
	CPPOPT_USING_PARAMS(Impl, ::nlpp::impl::Newton<::nlpp::poly::GradientOptimizer<V>, Factorization>);
	using Impl::Impl;

	virtual V optimize (::nlpp::wrap::poly::FunctionGradient<V> f, ::nlpp::wrap::poly::Hessian<V> hess, V x)
	{
		return Impl::optimize(f, hess, x);
	}

	virtual V optimize (::nlpp::wrap::poly::FunctionGradient<V> f, V x)
	{
		return optimize(f, ::nlpp::wrap::poly::Hessian<V>(::nlpp::fd::hessian(f.func)), x);
	}


	virtual Newton* clone_impl () const { return new Newton(*this); }
};

} // namespace poly





} // namespace nlpp
