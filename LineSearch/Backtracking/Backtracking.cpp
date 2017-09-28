#include "Goldstein.h"

#include "../../GradientDescent/GradientDescent.h"

#include "../../Newton/Newton.h"


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
	Goldstein ls(2.999);

	double x = 1.5;

	x += ls(bowl, x);

	DB(x);



	// GradientDescent<Goldstein> gd(Goldstein(1.0));

	// gd.maxIterations = 1e2;

	// Vec x(2); x << 10.0, 10.0;

	// x = gd(Bowl{}, x);

	// DB(x);



	// Newton<Goldstein> nm(Goldstein(1.0));


	// Vec x = Vec::Constant(10, 2.0);


	// x = nm(Rosenbrock(), x);

	// DB(x.transpose() << "         " << Rosenbrock()(x) << "\n");

	// DB(gradientFD(Rosenbrock())(x).transpose());




	return 0;
}
