#include "Newton/Newton.h"

#include "../LineSearch/Goldstein/Goldstein.h"

#include "../LineSearch/Backtracking/Backtracking.h"


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
	Newton<StrongWolfe, CholeskyIdentity> newton(StrongWolfe(1.0, 0.2));;


	Vec x = Vec::Constant(500, 5.0);

	x = newton(Rosenbrock(), x);


	DB(x.transpose());




	return 0;
}