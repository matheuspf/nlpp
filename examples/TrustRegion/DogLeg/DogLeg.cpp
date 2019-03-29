#include "TrustRegion/DogLeg/DogLeg.h"

#include "TestFunctions/Rosenbrock.h"

using namespace nlpp;



int main ()
{
	DogLeg<> opt;

	Vec x = Vec::Constant(100, 2.0);

	x = opt(Rosenbrock{}, x);

	handy::print(x.transpose());


	return 0;
}