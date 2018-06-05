#include "QuasiNewton/BFGS/BFGS.h"

#include "TestFunctions/Rosenbrock.h"


using namespace nlpp;


int main ()
{
    using LS = StrongWolfe;

    params::BFGS<LS> params;

    params.maxIterations = 1e4;
    params.gTol = 1e-8;
    params.fTol = 1e-8;
    params.xTol = 1e-8;


    BFGS<LS> bfgs(params);


    Vec x = Vec::Constant(10, 5.0);

    //Vec x(2); x << -1.2, 1;


    handy::benchmark([&]
    {
        x = bfgs(Rosenbrock(), x);
    });

    handy::print(x.transpose());


    return 0;
}