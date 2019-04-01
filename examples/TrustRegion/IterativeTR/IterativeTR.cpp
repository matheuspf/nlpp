#include "TrustRegion/IterativeTR/IterativeTR.h"

#include "TestFunctions/Rosenbrock.h"

using namespace nlpp;



int main ()
{
	IterativeTR<> opt;

	Vec x = Vec::Constant(1000, 5.0);

	x = opt(Rosenbrock{}, x);

	handy::print(x.transpose());


	return 0;
}