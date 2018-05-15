#include "Newton/Newton.h"

#include "LineSearch/Goldstein/Goldstein.h"

#include "LineSearch/Backtracking/Backtracking.h"

#include "TestFunctions/Rosenbrock.h"


using namespace cppnlp;



int main ()
{
	Newton<StrongWolfe, CholeskyIdentity> newton(StrongWolfe(1.0, 0.2));


	Vec x = Vec::Constant(50, 5.0);
	// Eigen::Matrix<double, 50, 1> x;
	// std::fill(&x(0), &x(0) + x.size(), 5.0);


	Rosenbrock func;

	handy::print(handy::benchmark([&]{
		x = newton(func, fd::gradient<fd::Backward>(func), fd::hessian<fd::Backward>(func), x);
	}), "\n\n");


	handy::print(x.transpose());




	return 0;
}