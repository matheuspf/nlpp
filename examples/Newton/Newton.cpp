#include "Newton/Newton.h"

#include "LineSearch/Goldstein/Goldstein.h"

#include "LineSearch/Backtracking/Backtracking.h"

#include "TestFunctions/Rosenbrock.h"


using namespace cppnlp;



int main ()
{
	Newton<StrongWolfe, CholeskyIdentity> newton(StrongWolfe(1.0, 0.2));;


	Vec x = Vec::Constant(500, 5.0);

	x = newton(Rosenbrock(), x);


	handy::print(x.transpose());




	return 0;
}