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


/// Macro aliases
#define CPPOPT_USING_PARAMS_CG(...) CPPOPT_USING_PARAMS(__VA_ARGS__);	\
									using Params::cg;					\
									using Params::v;


/// Builds a functor, to avoid a lot of copy/paste
#define BUILD_CG_STRUCT(Op, ...) \
\
struct __VA_ARGS__ \
{\
	template <class V>	\
	impl::Scalar<V> operator () (const V& fa, const V& fb, const V& dir = V()) const \
	{\
		Op; \
	}\
};



namespace nlpp
{

/** @name
 *  @brief The choice of the factor
*/
//@{
BUILD_CG_STRUCT(return fb.dot(fb) / fa.dot(fa), FR);

BUILD_CG_STRUCT(return fb.dot(fb - fa) / fa.dot(fa), PR);

BUILD_CG_STRUCT(return std::abs(PR::operator()(fa, fb)), PR_Abs : PR);

BUILD_CG_STRUCT(return std::max(0.0, PR::operator()(fa, fb)), PR_Plus : PR);

BUILD_CG_STRUCT(return fb.dot(fb - fa) / dir.dot(fb - fa), HS);

BUILD_CG_STRUCT(return fb.dot(fb) / dir.dot(fb - fa), DY);

BUILD_CG_STRUCT(Vec y = fb - fa;

				auto yp = y.dot(dir);
				auto yy = y.dot(y);

				return (y - 2 * dir * (yy / yp)).dot(fb) / yp;,

				HZ);

BUILD_CG_STRUCT(auto fr = FR::operator()(fa, fb);
				auto pr = PR::operator()(fa, fb);

				if(pr < -fr) return -fr;

				if(std::abs(pr) <= fr) return pr;

				return fr;,

				FR_PR : FR, PR);
//@}



namespace impl
{

namespace params
{

/** @brief Conjugate gradient parameters class, extending flow base GradientOptimizer parameters
*/
template <class Params_, class CGType = FR_PR>
struct CG : public Params_
{
	CPPOPT_USING_PARAMS(Params, Params_);

	using LineSearch = typename Params::LineSearch;
	using Stop = typename Params::Stop;
	using Output = typename Params::Output;

	CG() {}

	template <class LS = std::decay_t<LineSearch>, std::enable_if_t<std::is_same<LS, ::nlpp::StrongWolfe<>>::value, int> = 0>
    CG(const LineSearch& lineSearch = ::nlpp::StrongWolfe<>(1e-2, 1e-4, 0.1), const Stop& stop = Stop{}, const Output& output = Output{}) :
       Params(lineSearch, stop, output)
    {
    }


	CGType cg;			///< The functor that calculates the search direction, given the current gradient and the previous directions

	double v = 0.1;		///< The minimum factor of orthogonality that the current direction must have
};


} // namespace params




template <class Params_, class CGType = FR_PR>
struct CG : public params::CG<Params_, CGType>
{
	CPPOPT_USING_PARAMS_CG(Params, params::CG<Params_, CGType>);
	using Params::Params;


	template <class Function, class V>
	V optimize (Function f, V x)
	{
		V fa, dir, fb(x.rows(), x.cols());

		impl::Scalar<V> fxOld, fx;

		std::tie(fxOld, fa) = f(x);

		dir = -fa;

		stop.init(*this, x, fxOld, fa);
		output.init(*this, x, fxOld, fa);


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

		output.finish(*this, x, fx, fb);

		return x;
	}
};

} // namespace impl

template <class CGType = FR_PR, class LineSearch = StrongWolfe<>, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<0>>
struct CG : public impl::CG<params::GradientOptimizer<LineSearch, Stop, Output>, CGType>,
			public GradientOptimizer<CG<CGType, LineSearch, Stop, Output>>
{
	CPPOPT_USING_PARAMS(Impl, impl::CG<params::GradientOptimizer<LineSearch, Stop, Output>, CGType>);
	using Impl::Impl;

	template <class Function, class V>
	V optimize (Function f, V x)
	{
		return Impl::optimize(f, x);
	}
};

namespace poly
{

template <class CGType = ::nlpp::FR_PR, class V = ::nlpp::Vec>
struct CG : public ::nlpp::impl::CG<::nlpp::poly::GradientOptimizer<V>, CGType>
{
	CPPOPT_USING_PARAMS(Impl, ::nlpp::impl::CG<::nlpp::poly::GradientOptimizer<V>, CGType>);
	using Impl::Impl;
	using Vec = V;

    CG () {}

	virtual V optimize (::nlpp::wrap::poly::FunctionGradient<V> f, V x)
	{
		return Impl::optimize(f, x);
	}

	virtual CG* clone_impl () const { return new CG(*this); }
};


} // namespace poly





} // namespace nlpp