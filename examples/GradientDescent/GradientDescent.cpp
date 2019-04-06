#include "GradientDescent/GradientDescent.h"

#include "TestFunctions/Rosenbrock.h"



int main ()
{
	nlpp::poly::GradientDescent<> opt(//std::make_unique<nlpp::poly::Goldstein<>>(), 
									  nlpp::poly::StrongWolfe<>(),
									  std::make_unique<nlpp::stop::poly::GradientOptimizer<0>>(),
									  std::make_unique<nlpp::out::poly::GradientOptimizer<1>>());

	nlpp::Vec x = nlpp::Vec::Constant(10, 1.2);

	nlpp::Rosenbrock func;

	auto res = opt(func, x);

	handy::print(res.transpose());



	return 0;
}