#include "LineSearch/Goldstein/Goldstein.hpp"

#include "GradientDescent/GradientDescent.h"

#include "Newton/Newton.h"


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
	Goldstein ls(3.0, 0.49);

	double x = 1.0;

	x += ls(bowl, x);

	handy::print(x, "    ", bowl(x));



	// GradientDescent<Goldstein> gd(Goldstein(1.0));

	// gd.maxIterations = 1e2;

	// Vec x(2); x << 10.0, 10.0;

	// x = gd(Bowl{}, x);

	// DB(x);



	// Newton<Goldstein> nm(Goldstein(1.0));


	// Vec x = Vec::Constant(10, 2.0);


	// x = nm(Rosenbrock(), x);

	// DB(x.transpose() << "         " << Rosenbrock()(x) << "\n");

	// DB(fd::gradient(Rosenbrock())(x).transpose());




	return 0;
}
