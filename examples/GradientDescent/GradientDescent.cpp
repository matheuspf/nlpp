#include "GradientDescent/GradientDescent.h"

#include "TestFunctions/Rosenbrock.h"

using namespace cppnlp;



int main ()
{
	using LS = Goldstein;

	params::GradientDescent<LS> params(LS(), 10000, 1e-8, 1e-8, 1e-8);

	GradientDescent<> gd(params);

	Eigen::Vector2d x(1.2, 1.2);

	Rosenbrock func;

	auto grad = fd::gradient(func);


	// x = gd(func, x);
	// x = gd(func, grad, x);
	// x = gd([&](const auto& x){ return std::make_pair(func(x), grad(x)); }, x);
	x = gd([&](const auto& x, auto& g) { g = grad(x); return func(x); }, x);

	handy::print(x.transpose());


	return 0;
}