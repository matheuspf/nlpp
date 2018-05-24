#include "CG/CG.h"

#include "TestFunctions/Rosenbrock.h"

using namespace cppnlp;


int main ()
{
	CG<PR_FR> cg;

	Eigen::VectorXd x = Eigen::VectorXd::Constant(10, 5.0);

	handy::print(handy::benchmark([&]
	{
		x = cg(Rosenbrock(), x);
	}), "\n");

	handy::print("x: ", x.transpose());




	return 0;
}