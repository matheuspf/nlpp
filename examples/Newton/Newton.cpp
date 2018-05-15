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

	GradientFD<Rosenbrock, BackwardDifference> grad(func);
	HessianFD<Rosenbrock, BackwardDifference> hess(func);



	handy::print(handy::benchmark([&]{
		x = newton(Rosenbrock(), grad, hess, x);
		//x = newton(Rosenbrock(), gradientFD(Rosenbrock()), hessianFD(gradientFD(Rosenbrock())), x);
	}), "\n\n");


	handy::print(x.transpose());




	return 0;
}