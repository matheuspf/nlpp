#include "TrustRegion/IndefiniteDogLeg/IndefiniteDogLeg.h"

#include "TestFunctions/Rosenbrock.h"

using namespace cppnlp;


int main ()
{
	IndefiniteDogLeg idl;

	idl.maxIter = 1e2;

	Vec x = Vec::Constant(100, 5.0);
	//Vec x(2); x << -1.0, 1.2;

	x = idl(Rosenbrock{}, x);

	handy::print(x.transpose());


	return 0;
}