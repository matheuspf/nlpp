#include "GradientDescent/GradientDescent.h"

#include "TestFunctions/Rosenbrock.h"

using namespace nlpp;



int main ()
{
	using LS = Goldstein;

	params::GradientDescent<LS> params(LS(), 10000, 1e-8, 1e-8, 1e-8);

	GradientDescent<> gd(params);

	Eigen::VectorXd x(2); x << 1.2, 1.2;

	Rosenbrock func;

	auto grad = fd::gradient(func);


	handy::print("tm: ", handy::benchmark([&]{
		x = gd(func, x);
		// x = gd(func, grad, x);
		// x = gd([&](const auto& x){ return std::make_pair(func(x), grad(x)); }, x);
		// x = gd([&](const auto& x, auto& g) { g = grad(x); return func(x); }, x);
	}), "\n");

	handy::print("fx: ", func(x), "\n\nx:", x.transpose(), "\n");



	return 0;
}