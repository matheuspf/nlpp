#pragma once

#include "../LineSearch.h"

#include "../InitialStep/Constant.h"


namespace nlpp
{

namespace impl
{

template <typename Float = types::Float, class InitialStep = ConstantStep<Float>>
struct Goldstein : public LineSearchBase<Float, InitialStep>
{
	using Base = LineSearchBase<Float, InitialStep>;
	using Base::f0;
	using Base::g0;
	using Base::initialStep;


	Goldstein (Float c = 0.2, Float rho1 = 0.5, Float rho2 = 1.5, const InitialStep& initialStep = InitialStep(),
			   Float aMin = constants::eps, int maxIter = 100) : 
			   mu1(c), mu2(1.0 - c), Base(initialStep), rho1(rho1), rho2(rho2), aMin(aMin), maxIter(maxIter)
	{
		// assert(a0 > 1e-5 && "a0 must be positive");
		assert(c < 0.5 && "c must be smaller than 0.5");
		assert(rho1 < 1.0 && "rho1 must be smaller than 1.0");
		assert(rho2 > 1.0 && "rho2 must be greater than 1.0");
		// assert(aMin < a0 && "a0 must be greater than aMin");
	}


	template <class Function>
	Float lineSearch (Function f)
	{		
		std::tie(f0, g0) = f(0.0);

		Float a0 = initialStep(f0, g0);
		Float a = a0, safeGuard = a0;


		int iter = 0;

		while(a > aMin && ++iter < maxIter)
		{
			Float fa, ga;

			std::tie(fa, ga) = f(a);

			if(fa > f0 + mu1 * a * g0)
			{
				a = a * rho1;
				continue;
			}

			safeGuard = a;

			if(fa < f0 + mu2 * a * g0)
			{
				a = a * rho2;

				continue;
			}

			break;
		}

		return iter < maxIter ? a : safeGuard;
	}


	Float mu1, mu2;
	Float rho1, rho2;
	Float aMin;
	int maxIter;
};

} // namespace impl




NLPP_LINE_SEARCH(Goldstein)


} // namespace nlpp