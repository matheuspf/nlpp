#include "TrustRegion/DogLeg/DogLeg.h"

#include "TestFunctions/Rosenbrock.h"

using namespace cppnlp;



int main ()
{
	DogLeg cp;

	cp.maxIter = 1e2;

	//Vec x = Vec::Constant(2, 5.0);
	Vec x(2); x << -1.0, 1.2;

	x = cp(Rosenbrock{}, x);

	handy::print(x.transpose());


	return 0;
}