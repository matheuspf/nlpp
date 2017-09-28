#ifndef OPT_GOLDSTEIN_LS_H
#define OPT_GOLDSTEIN_LS_H

#include "../LineSearch.h"



struct Goldstein : public LineSearch<Goldstein>
{
	Goldstein (double a0 = 1.0, double c = 0.2, double rho1 = 0.5,
			   double rho2 = 1.5, double aMin = EPS, int maxIter = 100) : 
			   a0(a0), mu1(c), mu2(1.0 - c), rho1(rho1), 
			   rho2(rho2), aMin(aMin), maxIter(maxIter)
	{
		assert(a0 > 1e-5 && "a0 must be positive");
		assert(c < 0.5 && "c must be smaller than 0.5");
		assert(rho1 < 1.0 && "rho1 must be smaller than 1.0");
		assert(rho2 > 1.0 && "rho2 must be greater than 1.0");
		assert(aMin < a0 && "a0 must be greater than aMin");
	}


	template <class Function, class Gradient>
	double lineSearch (Function f, Gradient g)
	{
		double a = a0;

		double f0 = f(0.0);
		double g0 = g(0.0);

		double safeGuard = a0;

		int iter = 0;


		while(a > aMin && ++iter < maxIter)
		{
			double fa = f(a);
			double ga = g(a);

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



#endif // OPT_GOLDSTEIN_LS_H