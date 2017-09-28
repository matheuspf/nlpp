#ifndef OPT_GRADIENT_DESCENT_H
#define OPT_GRADIENT_DESCENT_H

#include "../../Modelo.h"

#include "../FiniteDifference.h"

#include "../LineSearch/Goldstein/Goldstein.h"




template <class LineSearch = Goldstein>
struct GradientDescent
{
	GradientDescent (LineSearch lineSearch = LineSearch{}) : lineSearch(lineSearch) {}


	template <class Function, class Gradient, typename Type>
	Type operator () (Function function, Gradient gradient, Type x)
	{
		Type direction = -gradient(x);

		Type ret = x;

		for(int iter = 0; iter < maxIterations && norm(direction) > xTol; ++iter)
		{
			double alpha = lineSearch(function, gradient, x, direction);

			x = x + alpha * direction;

			if(function(x) < function(ret))
				ret = x;

			direction = -gradient(x);
		}

		return ret;
	}


	template <class Function, typename Type>
	Type operator () (Function function, const Type& x)
	{
		return this->operator()(function, gradientFD(function), x);
	}



	LineSearch lineSearch;


	int maxIterations = 1e3;

	double xTol = 1e-6;

	double fTol = 1e-7;
};




#endif // endif OPT_GRADIENT_DESCENT_H