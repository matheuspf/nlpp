#include "CG/CG.h"

#include "TestFunctions/Rosenbrock.h"

using namespace cppnlp;


int main ()
{
	CG<PR_FR> cg;

	Vec x = Vec::Constant(100, 5.0);

	x = cg(Rosenbrock(), x);


	handy::print(x.transpose());




	return 0;
}