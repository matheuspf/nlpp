#include "GradientDescent/GradientDescent.h"

#include "TestFunctions/Rosenbrock.h"

using namespace cppnlp;


int main ()
{
	GradientDescent<> gd;

	Vec x = Vec::Constant(2, 1.2);

	x = gd(Rosenbrock{}, x);

	handy::print(x.transpose());


	return 0;
}