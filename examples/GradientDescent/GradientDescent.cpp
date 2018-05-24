#include "GradientDescent/GradientDescent.h"

#include "TestFunctions/Rosenbrock.h"

using namespace cppnlp;


int main ()
{
	GradientDescent<> gd;

	Eigen::Vector3d x(1.2, 1.2, 1.2);

	Rosenbrock func;

	auto grad = fd::gradient(func);

	//x = gd(func, x);
	//x = gd([&](const auto& x){ return std::make_pair(func(x), grad(x)); }, x);
	x = gd([&](const Eigen::Vector3d& x, Eigen::Vector3d& g){ g = grad(x); return func(x); }, x);

	handy::print(x.transpose());


	return 0;
}