#include "LineSearch/StrongWolfe/StrongWolfe.h"

#include "Newton/Newton.h"

#include "LineSearch/Backtracking/Backtracking.h"

#include "LineSearch/Goldstein/Goldstein.h"

#include "TestFunctions/Rosenbrock.h"


using namespace nlpp;


double senoid (double x)
{
	return sin(x*x);
}

double bowl (double x)
{
	return pow(x - 3, 2);
}


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


	Newton<StrongWolfe, fact::CholeskyIdentity> newton;
	//Newton<StrongWolfe, fact::CholeskyIdentity> newton(StrongWolfe(1.0, 0.2));


	Vec x = Vec::Constant(100, 5.0);

	x = newton(Rosenbrock(), x);


	handy::print(x.transpose());



	return 0;
}
