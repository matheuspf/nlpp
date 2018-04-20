#include "Newton/Newton.h"

#include "LineSearch/Goldstein/Goldstein.h"

#include "LineSearch/Backtracking/Backtracking.h"

#include "TestFunctions/Rosenbrock.h"


using namespace cppnlp;



int main ()
{
	Newton<StrongWolfe, CholeskyIdentity> newton(StrongWolfe(1.0, 0.2));


	Vec x = Vec::Constant(50, 5.0);

	handy::print(handy::benchmark([&]{
		x = newton(Rosenbrock(), x);
	}), "\n\n");


	handy::print(x.transpose());




	return 0;
}