#pragma once

#include "../Helpers/Helpers.h"

#include "../Helpers/FiniteDifference.h"

#include "../LineSearch/Goldstein/Goldstein.h"


namespace cppnlp
{

template <class LineSearch = Goldstein>
struct GradientDescent
{
	GradientDescent (LineSearch lineSearch = LineSearch{}) : lineSearch(lineSearch) {}


	template <class Function, class Gradient, typename Type>
	Type operator () (Function function, Gradient gradient, Type x)
	{
		Type direction = -gradient(x);

		Type ret = x;

		for(int iter = 0; iter < maxIterations && direction.norm() > xTol; ++iter)
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
		return this->operator()(function, fd::gradient(function), x);
	}



	LineSearch lineSearch;


	int maxIterations = 1e3;

	double xTol = 1e-6;

	double fTol = 1e-7;
};

} // namespace cppnlp
