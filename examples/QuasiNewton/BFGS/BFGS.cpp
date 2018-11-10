#include "QuasiNewton/BFGS/BFGS.h"

#include "LineSearch/Goldstein/Goldstein.h"

#include "TestFunctions/Rosenbrock.h"


using namespace nlpp;


int main ()
{
    using IH = BFGS_Diagonal;
    using LS = StrongWolfe;

    params::BFGS<IH, LS> params;

    params.stop.maxIterations = 1e4;
    params.stop.gTol = 1e-8;
    params.stop.fTol = 1e-8;
    params.stop.xTol = 1e-8;
    params.lineSearch = StrongWolfe(1.0, 1e-2, 0.1);


    BFGS<IH, LS> bfgs(params);


    Vec x = Vec::Constant(10, 5.0);

    //Vec x(2); x << -1.2, 1;


    handy::benchmark([&]
    {
        x = bfgs(Rosenbrock(), x);
    });

    handy::print(x.transpose());


    return 0;
}