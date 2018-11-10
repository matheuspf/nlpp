#include "GradientDescent/GradientDescent.h"

#include "TestFunctions/Rosenbrock.h"

using namespace nlpp;



int main ()
{
	using LS = Goldstein;
	using Stop = stop::GradientOptimizer<>;
	using Out = out::GradientOptimizer<1>;

	params::GradientDescent<LS, Stop, Out> params(LS{}, Stop(1000, 1e-6, 1e-6, 1e-6), Out{});

	GradientDescent<LS, Stop, Out> gd(params);


	Eigen::VectorXf x(2);
	x << 1.2, 1.2;

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