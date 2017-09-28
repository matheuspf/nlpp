#include "GradientDescent.h"


struct Rosenbrock
{
	double operator () (const Vec& x) const
	{
		double r = 0.0;

        for(int i = 0; i < x.rows() - 1; ++i)
        	r += 100.0 * pow(x(i+1) - pow(x(i), 2), 2) + pow(x(i) - 1.0, 2);

        return r;
	}
};



int main ()
{
	GradientDescent<> gd;

	Vec x = Vec::Constant(2, 1.2);

	x = gd(Rosenbrock{}, x);

	DB(x);


	return 0;
}