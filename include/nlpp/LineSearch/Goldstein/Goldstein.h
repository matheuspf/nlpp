#pragma once

#include "../LineSearch.h"


namespace nlpp
{

namespace impl
{

struct Goldstein
{
	Goldstein (double a0 = 1.0, double c = 0.2, double rho1 = 0.5, double rho2 = 1.5, 
			   double aMin = constants::eps, int maxIter = 100) : 
			   a0(a0), mu1(c), mu2(1.0 - c), rho1(rho1), rho2(rho2), aMin(aMin), maxIter(maxIter)
	{
		assert(a0 > 1e-5 && "a0 must be positive");
		assert(c < 0.5 && "c must be smaller than 0.5");
		assert(rho1 < 1.0 && "rho1 must be smaller than 1.0");
		assert(rho2 > 1.0 && "rho2 must be greater than 1.0");
		assert(aMin < a0 && "a0 must be greater than aMin");
	}


	template <class Function>
	double lineSearch (Function f)
	{
		double f0, g0, a = a0, safeGuard = a0;
		
		std::tie(f0, g0) = f(0.0);


		int iter = 0;

		while(a > aMin && ++iter < maxIter)
		{
			double fa, ga;

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


	double a0;
	double mu1, mu2;
	double rho1, rho2;
	double aMin;
	int maxIter;
};

} // namespace impl


struct Goldstein : public impl::Goldstein,
				   public LineSearch<Goldstein>
{
	using impl::Goldstein::Goldstein;

	template <class Function>
	double lineSearch (Function f)
	{
		return impl::Goldstein::lineSearch(f);
	}
};


namespace poly
{

template <class Function>
struct Goldstein : public impl::Goldstein,
				   public LineSearch<Function>
{
	using impl::Goldstein::Goldstein;

	double lineSearch (Function f)
	{
		return impl::Goldstein::lineSearch(f);
	}

	Goldstein* clone () const
	{
		return new Goldstein(*this);
	}
};

} // namespace poly




} // namespace nlpp