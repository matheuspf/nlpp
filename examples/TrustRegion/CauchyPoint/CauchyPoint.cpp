#include "TrustRegion/CauchyPoint/CauchyPoint.h"

#include "TestFunctions/Rosenbrock.h"

using namespace nlpp;




int main ()
{
	CauchyPoint<> opt;

	Vec x = Vec::Constant(2, 1.2);

	x = opt(Rosenbrock{}, x);

	handy::print(x.transpose());


	return 0;
}