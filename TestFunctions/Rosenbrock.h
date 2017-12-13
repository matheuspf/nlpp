#ifndef OPT_TEST_FUNCTIONS_H
#define OPT_TEST_FUNCTIONS_H

#include "../Modelo.h"


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


#endif // OPT_TEST_FUNCTIONS_H