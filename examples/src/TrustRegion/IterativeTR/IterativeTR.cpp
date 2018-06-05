#include "TrustRegion/IterativeTR/IterativeTR.h"

#include "TestFunctions/Rosenbrock.h"

using namespace nlpp;



int main ()
{
	IterativeTR idl;

    idl.maxIter = 1e3;
    //idl.delta0 = 50;

	Vec x = Vec::Constant(10, 5.0);
	//Vec x(2); x << -1.0, 1.2;

	x = idl(Rosenbrock{}, x);

	handy::print(x.transpose());


	return 0;
}