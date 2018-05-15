#pragma once

#include "../Helpers/Helpers.h"

#include "../LineSearch/Goldstein/Goldstein.h"

#include "../LineSearch/ConstantStep/ConstantStep.h"

#include "../LineSearch/StrongWolfe/StrongWolfe.h"

#include "Factorizations.h"


namespace cppnlp
{

template <class LineSearch = ConstantStep, class Inversion = SmallIdentity>
struct Newton
{
	Newton (LineSearch lineSearch = LineSearch{}, Inversion inversion = Inversion{}) :
			lineSearch(lineSearch), inversion(inversion) {}


	template <class Function, class Gradient, class Hessian, typename Type>
	Type operator () (Function function, Gradient gradient, Hessian hessian, Type x)
	{
		Type dir = direction(gradient(x), hessian(x));


		for(int iter = 0; iter < maxIterations && dir.norm() > xTol; ++iter)
		{
			double alpha = lineSearch(function, gradient, x, dir);

			x = x + alpha * dir;

			dir = direction(gradient(x), hessian(x));
		}

		return x;
	}


	template <class Function, class Gradient, typename Type>
	Type operator () (Function function, Gradient gradient, Type x)
	{
		return this->operator()(function, gradient, fd::hessian(function), x);
	}

	template <class Function, typename Type>
	Type operator () (Function function, Type x)
	{
		return this->operator()(function, fd::gradient(function), x);
	}




	double direction (double gx, double hx)
	{
		if(abs(hx) < 1e-8)
		{
			handy::print("Second derivative is very close to 0. Making a small correction.\n");

			hx = 1e-5;
		}


		return -gx / hx;
	}




	Vec direction (const Vec& grad, const Mat& hess)
	{
		return Inversion(inversion)(grad, hess);
	}




	LineSearch lineSearch;

	Inversion inversion;


	int maxIterations = 1e3;

	double xTol = 1e-6;

	double fTol = 1e-7;

};

} // namespace cppnlp

