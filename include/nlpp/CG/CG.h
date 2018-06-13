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
	double operator () (const Vec& fa, const Vec& fb, const Vec& dir = Vec()) const \
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

				double yp = y.dot(dir);
				double yy = y.dot(y);

				return (y - 2 * dir * (yy / yp)).dot(fb) / yp;,

				HZ);

BUILD_CG_STRUCT(double fr = FR::operator()(fa, fb);
				double pr = PR::operator()(fa, fb);

				if(pr < -fr) return -fr;

				if(std::abs(pr) <= fr) return pr;

				return fr;,

				FR_PR : FR, PR);
//@}





namespace params
{


template <class CGType = FR_PR, class LineSearch = StrongWolfe, class Output = out::GradientOptimizer<0>>
struct CG : public GradientOptimizer<LineSearch, Output>
{
	using Params = GradientOptimizer<LineSearch, Output>;

	template <typename... Args, class LS = std::decay_t<LineSearch>, std::enable_if_t<std::is_same<LS, StrongWolfe>::value, int> = 0>
	CG(const LineSearch& lineSearch = StrongWolfe(1e-2, 1e-4, 0.1), Args&&... args) : Params(lineSearch, std::forward<Args>(args)...)
	{	
	}

	template <typename... Args, class LS = std::decay_t<LineSearch>, std::enable_if_t<!std::is_same<LS, StrongWolfe>::value, int> = 0>
	CG(const LineSearch& lineSearch = LineSearch{}, Args&&... args) : Params(lineSearch, std::forward<Args>(args)...) 
	{
	}


	CGType cg;

	double v = 0.1;
};


} // namespace params



template <class CGType = FR_PR, class LineSearch = StrongWolfe, class Output = out::GradientOptimizer<0>>
struct CG : public GradientOptimizer<CG<CGType, LineSearch, Output>, params::CG<CGType, LineSearch, Output>>
{
	CPPOPT_USING_PARAMS_CG(Params, GradientOptimizer<CG<CGType, LineSearch, Output>, params::CG<CGType, LineSearch, Output>>);

	using Params::Params;


	template <class Function, class Float, int Rows, int Cols>
	auto optimize (Function f, Eigen::Matrix<Float, Rows, Cols> x)
	{
		Eigen::Matrix<Float, Rows, Cols> fa, dir, fb(x.rows(), x.cols());

		Float fxOld;

		std::tie(fxOld, fa) = f(x);

		dir = -fa;


		for(int iter = 0; iter < maxIterations && dir.norm() > xTol; ++iter)
		{
			double alpha = lineSearch(f, x, dir);
			
			x = x + alpha * dir;

			Float fx = f(x, fb);
			

			if(std::abs(fxOld - fx) < fTol)
				break;

			if((fa.dot(fb) / fb.dot(fb)) >= v)
				dir = -fb;

			else
				dir = -fb + cg(fa, fb, dir) * dir;


			fxOld = fx;

			fa = fb;
		}

		return x;
	}
};


} // namespace nlpp