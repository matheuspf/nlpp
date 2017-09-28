#include "BFGS.h"

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
    BFGS<> bfgs;

    bfgs.maxIter = 1e3;
    bfgs.gTol = 1e-4;


    Vec x = Vec::Constant(500, 5.0);

    //Vec x(2); x << -1.2, 1;


    benchmark([&]
    {
        x = bfgs(Rosenbrock(), x);
    });

    DB(x.transpose());


    return 0;
}