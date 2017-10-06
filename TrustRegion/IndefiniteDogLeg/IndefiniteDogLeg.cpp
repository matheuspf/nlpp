#include "IndefiniteDogLeg.h"


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
	IndefiniteDogLeg idl;

	idl.maxIter = 1e2;

	Vec x = Vec::Constant(2, 5.0);
	//Vec x(2); x << -1.0, 1.2;

	x = idl(Rosenbrock{}, x);

	DB(x.transpose());


	return 0;
}