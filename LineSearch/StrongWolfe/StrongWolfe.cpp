#include "StrongWolfe.h"

// #include "../../GradientDescent/GradientDescent.h"

#include "../../Newton/Newton.h"

#include "../Backtracking/Backtracking.h"

#include "../Goldstein/Goldstein.h"


double senoid (double x)
{
	return sin(x*x);
}

double bowl (double x)
{
	return pow(x - 3, 2);
}


struct Rosenbrock
{
	double operator () (const VectorXd& x) const
	{
		double r = 0.0;

        for(int i = 0; i < x.rows() - 1; ++i)
        	r += 100.0 * pow(x(i+1) - pow(x(i), 2), 2) + pow(x(i) - 1.0, 2);

        return r;
	}
};


struct Bowl
{
	Bowl (Vec c = Vec::Constant(2, 0.0)) : c(c) {}

	double operator () (const Vec& x) const
	{
		return (x - c).norm();
	}

	Vec c;
};




int main ()
{
	// StrongWolfe ls;

	// double a = 0.0;

	// FOR(i, 10)
	// {
	// 	a += ls(Rosenbrock(), Vec::Constant(2, -1.3 + a), Vec::Constant(2, 1.0));

	// 	DB(a);
	// }


	Newton<StrongWolfe, CholeskyIdentity> newton(StrongWolfe(1.0, 0.2));


	Vec x = Vec::Constant(100, 5.0);

	x = newton(Rosenbrock(), x);


	DB(x.transpose());



	return 0;
}
