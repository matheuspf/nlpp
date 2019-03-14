#include "TrustRegion/DogLeg/DogLeg.h"

#include "TestFunctions/Rosenbrock.h"

using namespace nlpp;



int main ()
{
	DogLeg cp;

	cp.maxIter = 1e2;

	Vec x = Vec::Constant(100, 2.0);
	// Vec x(2); x << -1.0, 1.2;
	// Vec x(10);

	// std::for_each(x.data(), x.data() + x.size(), [](auto& xi){ xi = handy::rand(-2.0, 2.0); });

	x = cp(Rosenbrock{}, x);

	handy::print(x.transpose());


	return 0;
}