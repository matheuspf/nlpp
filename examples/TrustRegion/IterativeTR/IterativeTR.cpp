#include "TrustRegion/IterativeTR/IterativeTR.h"

#include "TestFunctions/Rosenbrock.h"

using namespace nlpp;



int main ()
{
	// IterativeTR<> opt;
	IterativeTR<stop::GradientNorm<>> opt;
    opt.stop = stop::GradientNorm<>(1e4, 1e-6);

	Vec x = Vec::Constant(50, 5.0);

	x = opt(Rosenbrock{}, x);

	auto gx = fd::gradient(Rosenbrock{})(x);

	handy::print(x.transpose());
	handy::print(opt.stop(opt, Vec(), 0.0, gx));

	return 0;
}
