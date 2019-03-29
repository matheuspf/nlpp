#include "TrustRegion/IndefiniteDogLeg/IndefiniteDogLeg.h"

#include "TestFunctions/Rosenbrock.h"

using namespace nlpp;


int main ()
{
	IndefiniteDogLeg<> opt;

	Vec x = Vec::Constant(100, 2.0);

	x = opt(Rosenbrock{}, x);

	handy::print(x.transpose());


	return 0;
}