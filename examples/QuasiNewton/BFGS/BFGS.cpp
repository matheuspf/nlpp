#include "QuasiNewton/BFGS/BFGS.h"

#include "TestFunctions/Rosenbrock.h"


using namespace cppnlp;


int main ()
{
    BFGS<> bfgs;

    bfgs.maxIter = 1e3;
    bfgs.gTol = 1e-4;


    Vec x = Vec::Constant(1000, 5.0);

    //Vec x(2); x << -1.2, 1;


    handy::benchmark([&]
    {
        x = bfgs(Rosenbrock(), x);
    });

    handy::print(x.transpose());


    return 0;
}