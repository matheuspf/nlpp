#pragma once

#include "../LineSearch.hpp"


namespace nlpp
{

/// Constant step line search
struct ConstantStep : public LineSearch<ConstantStep>
{
	/// Initial step and reduction factor
	ConstantStep (double a0 = 1.0, double rho = 1.0) : a0(a0), rho(rho)
	{
		assert(a0 > 1e-5 && "a0 must not be so small");
		assert(rho <= 1.0 && "rho must be no greater than 1.0");
	}


	/// Simply returns a constant, reducing the initial step @c a0 by rho at each step
	double lineSearch (...)
	{
		double a = a0;

		a0 *= rho;

		return a;
	}


	double a0;		///< Initial step

	double rho;		///< Reduction factor
};

}// namespace nlpp
