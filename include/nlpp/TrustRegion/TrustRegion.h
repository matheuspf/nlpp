#pragma once

#include "../Helpers/Helpers.h"
#include "../Helpers/FiniteDifference.h"
#include "../Helpers/Parameters.h"
#include "../Helpers/Optimizer.h"



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

	TrustRegion (const LocalOptimizer& localOptimizer, Float delta0 = 10.0, Float alpha = 0.25, Float beta = 2.0, Float eta = 0.1, Float maxDelta = 1e2) :
				 localOptimizer(localOptimizer), delta0(delta0), alpha(alpha), beta(beta), eta(eta), maxDelta(maxDelta) {}


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
struct TrustRegion : public ::nlpp::params::TrustRegion<LocalOptimizer, Params_, Float>
{
	using Params = ::nlpp::params::TrustRegion<LocalOptimizer, Params_, Float>;
	using Params::Params;
    using Params::localOptimizer;
    using Params::stop;
    using Params::alpha;
    using Params::output;
    using Params::delta0;
    using Params::maxDelta;
    using Params::beta;
    using Params::eta;


	template <class Function, class Hessian, class V>
	V optimize (Function function, Hessian hessian, V x)
	{
		Float delta = delta0;
        
        Float fx, fxp;
        V p, gx, gxp;

        std::tie(fx, gx) = function(x);

		Mat hx = hessian(x);


		for(int iter = 0; iter < stop.maxIterations(); ++iter)
		{
			/** Making a call to the actual function that generates the direction whitin the trust region.
			  *	I am using CRTP here, so the 'Impl' class inherits from this class. **/
			std::tie(p, fxp, gxp) = localOptimizer(function, hessian, x, gx, hx, delta);

			Float aRed = (fx - fxp);	/// Actual reduction 

			Float pRed = -(gx.dot(p) + 0.5 * p.transpose() * hx * p);	/// Predicted reduction

			/** Ratio between actual and predicted reduction. 'pRed' SHOULD always be positive, but I am also 
			 *  handling the case where the trust region direction fails miserably and returns a worst solution.
			 *  That 'abs' actually should not be there.
			**/
			Float rho = aRed / std::max(constants::eps_<Float>, std::abs(pRed));

			//db(delta, "       ", function(x), "      ", aRed, "      ", pRed, "       ", rho, "       ", x.transpose(), "         ", dir.transpose(), "\n\n\n");
			//db((fx - fy), "      ", (-gx.dot(dir) - 0.5 * dir.transpose() * hx * dir), "       ", x.transpose(), "     ", dir.transpose(), "\n\n\n");

			if(std::isnan(rho))
			{
				handy::print("FAIL HARD");
				handy::print("\n\n\n", fx, "     ", fxp, "          ", gx.transpose(), "          ", p.transpose(), "\n\n");
				std::cout << hx << "\n\n";
                exit(0);
			}



			/// If this is the maximum allowed value for delta and we had this small improvement, theres nothing else to do
			if(aRed < constants::eps_<Float> && delta == maxDelta)
				return x;

			/// If rho is smaller than this threshold, reduce the trust region size
			if(rho < alpha)
				delta = alpha * delta;

			/** If rho is greater than this other threshold AND the direction is at
			  *	the border of the trust region, then increase the trust region size.
			  *
			  * TODO: change that aproximation (1e-4) and return a flag telling if the direction is at the border.
			**/
			else if(rho > 1.0 - alpha && std::pow(p.norm() - delta, 2) < constants::eps_<Float>)
				delta = std::min(beta * delta, maxDelta);


			/// Too small trust region, go home. Nothing else to do
			if(std::pow(delta, 2) < 2 * constants::eps_<Float>) 
				return x;

			// if(stop(*this, x, fx, gx))
            //     return x;

			/// We only update the actual x if rho is greater than eta and the function value is actually reduced
			if(rho > eta)
			{
                std::tie(x, fx, gx) = std::tie(x + p, fxp, gxp);
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

template <class V = ::nlpp::Vec, class M = ::nlpp::Mat>
struct LocalMinimizerBase : public ::nlpp::poly::CloneBase<LocalMinimizerBase<V>>
{
	using Float = ::nlpp::impl::Scalar<V>;

	virtual ~LocalMinimizerBase() {}

	virtual std::tuple<V, Float, V> operator () (::nlpp::wrap::poly::FunctionGradient<V>, ::nlpp::wrap::poly::Hessian<V, M>, const V&, const V&, const M&, Float) = 0;
};


template <class V = ::nlpp::Vec, class M = ::nlpp::Mat>
struct LocalMinimizer_ : public ::nlpp::poly::PolyClass<LocalMinimizer_<V, M>>
{
    NLPP_USING_POLY_CLASS(Base, ::nlpp::poly::PolyClass<LocalMinimizer_<V, M>>);

    // LocalMinimizer_ () : Base(std::make_unique<CauchyPoint<>>()) {}

    using Float = ::nlpp::impl::Scalar<V>;

	std::tuple<V, Float, V> operator () (::nlpp::wrap::poly::FunctionGradient<V> function, ::nlpp::wrap::poly::Hessian<V, M> gradient,
										 const V& x, const V& gx, const M& hx, Float delta)
	{
		return impl->operator()(function, gradient, x, gx, hx, delta);
	}
};



template <class V = ::nlpp::Vec, class M = ::nlpp::Mat>
struct TrustRegion : public ::nlpp::impl::TrustRegion<LocalMinimizer_<V, M>, ::nlpp::params::poly::Optimizer_>
{
	using Impl = ::nlpp::impl::TrustRegion<LocalMinimizer_<V, M>, ::nlpp::params::poly::Optimizer_>;
	using Impl::Impl;

	virtual V optimize (::nlpp::wrap::poly::FunctionGradient<V> f, ::nlpp::wrap::poly::Hessian<V, M> hess, V x)
	{
		return Impl::optimize(f, hess, x);
	}

	virtual V optimize (::nlpp::wrap::poly::FunctionGradient<V> f, V x)
	{
		return optimize(f, ::nlpp::wrap::poly::Hessian<V, M>(::nlpp::fd::hessian(f.func)), x);
	}


	virtual TrustRegion* clone_impl () const { return new TrustRegion(*this); }
};

} // namespace poly




} // namespace nlpp
