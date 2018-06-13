#include "QuasiNewton/LBFGS/LBFGS.h"

#include "LineSearch/Dynamic/Dynamic.h"

#include "LineSearch/StrongWolfe/StrongWolfe.h"

#include "LineSearch/Goldstein/Goldstein.h"

#include "TestFunctions/Rosenbrock.h"


using namespace nlpp;


int main ()
{
    using Func = Rosenbrock;
    using IH = BFGS_Diagonal;
    using LS = StrongWolfe;
    using Out = out::GradientOptimizer<0>;

    params::LBFGS<IH, LS, Out> params;

    params.maxIterations = 1e4;
    params.fTol = 1e-8;
    params.gTol = 1e-8;
    params.xTol = 1e-8;
    params.m = 10;

    //params.lineSearch = StrongWolfe(1.0, 1e-4, 0.1);

    LBFGS<Vec, IH, LS, Out> lbfgs(params);

    Func func;
    Vec x = Vec::Constant(2, 1.2);

    handy::benchmark([&]
    {
        x = lbfgs(func, x);
    });

    handy::print(x.transpose());


    return 0;
}