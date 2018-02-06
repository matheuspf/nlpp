#pragma once

#include "../LineSearch.h"


namespace cppnl
{

struct ConstantStep : public LineSearch<ConstantStep>
{
	ConstantStep (double a0 = 1.0, double rho = 1.0) : a0(a0), rho(rho)
	{
		assert(a0 > 1e-5 && "a0 must not be so small");
		assert(rho <= 1.0 && "rho must be no greater than 1.0");
	}


	double lineSearch (...)
	{
		double a = a0;

		a0 *= rho;

		return a;
	}


	double a0;
	double rho;
};

}// namespace cppnl
