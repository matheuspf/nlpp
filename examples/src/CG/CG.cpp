#include "CG/CG.h"

#include "TestFunctions/Rosenbrock.h"

#include "LineSearch/Goldstein/Goldstein.h"



using namespace nlpp;


int main ()
{
	params::CG<FR_PR, StrongWolfe> params(StrongWolfe{}, stop::GradientOptimizer{});

	params.stop.fTol = 0.0;

	CG<FR_PR, StrongWolfe> cg(StrongWolfe{}, stop::GradientOptimizer{});

	Rosenbrock func;

	auto grad = fd::gradient(func);
	//auto grad = [](const auto& x){ return x; };

	Eigen::VectorXd x = Eigen::VectorXd::Constant(10, 5.0);
	
	handy::print(handy::benchmark([&]
	{
		//x = cg(func, x);
		// x = cg(func, grad, x);
		x = cg([&](const auto& x){ return std::make_pair(func(x), grad(x)); }, x);
		//x = cg([&](const auto& x, auto& g){ g = grad(x); return func(x); }, x);
	}), "\n");

	handy::print("x: ", x.transpose());




	return 0;
}