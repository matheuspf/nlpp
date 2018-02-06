#include "TrustRegion/CauchyPoint/CauchyPoint.h"

#include "TestFunctions/Rosenbrock.h"

using namespace cppnlp;




int main ()
{
	CauchyPoint cp;

	Vec x = Vec::Constant(2, 1.2);

	x = cp(Rosenbrock{}, x);

	handy::print(x.transpose());


	return 0;
}