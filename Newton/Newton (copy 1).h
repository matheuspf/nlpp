#ifndef OPT_NEWTON_H
#define OPT_NEWTON_H

#include "../../Modelo.h"

//#include "../LineSearch/Goldstein/Goldstein.h"

#include "../LineSearch/ConstantStep/ConstantStep.h"

#include "../Parameters.h"




template <class LineSearch = ConstantStep>
struct Newton
{
	Newton (LineSearch lineSearch = LineSearch{}) : lineSearch(lineSearch) {}



	template <class Function, class Gradient, class Hessian, typename Type>
	Type operator () (Function function, Gradient gradient, Hessian hessian, Type x)
	{
		Type dir = direction(hessian(x), gradient(x));

		//DB("x:   " << x.transpose() << "        dir:  " << dir.transpose() << "\n\n");

		for(int iter = 0; iter < maxIterations && norm(dir) > xTol; ++iter)
		{
			double alpha = lineSearch(function, gradient, x, dir);

			x = x + alpha * dir;

			dir = direction(hessian(x), gradient(x));

			//DB("x:   " << x.transpose() << "        dir:  " << dir.transpose() << "\n\n");
		}

		return x;
	}


	template <class Function, class Gradient, typename Type>
	Type operator () (Function function, Gradient gradient, Type x)
	{
		return this->operator()(function, gradient, hessianFD(function), x);
	}

	template <class Function, typename Type>
	Type operator () (Function function, Type x)
	{
		return this->operator()(function, gradientFD(function), x);
	}



	VectorXd direction (const MatrixXd& h, const VectorXd& b)
	{
		return -h.colPivHouseholderQr().solve(b);
	}


	double direction (double h, double b)
	{
		return -b / h;
	}




	LineSearch lineSearch;


	int maxIterations = 1e2;

	double xTol = 1e-6;

	double fTol = 1e-7;

};


#endif // endif OPT_NEWTON_H