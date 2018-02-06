#pragma once

#include "../LineSearch.h"


namespace cppnl
{

struct Backtracking : public LineSearch<Backtracking>
{
	Backtracking (double a0 = 1.0, double c = 1e-4, double rho = 0.5, double aMin = constants::eps) :
				  a0(a0), c(c), rho(rho), aMin(aMin)
	{
		assert(a0 > 1e-5 && "a0 must be positive");
		assert(c < 1.0 && "c must be smaller than 1.0");
		assert(rho < 1.0 && "rho must be smaller than 1.0");
		assert(aMin < a0 && "a0 must be greater than aMin");
	}


	template <class Function, class Gradient>
	double lineSearch (Function f, Gradient g)
	{
		double a = a0;

		double f0 = f(0.0);
		double g0 = g(0.0);

		while(a > aMin && f(a) > f0 + c * a * g0)
			a = rho * a;

		return a;
	}


	double a0;
	double c;
	double aMin;
	double rho;
};

} // namespace cppnl