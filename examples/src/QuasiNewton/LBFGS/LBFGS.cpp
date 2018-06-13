#include "QuasiNewton/LBFGS/LBFGS.h"

#include "LineSearch/Dynamic/Dynamic.h"

#include "TestFunctions/Rosenbrock.h"


using namespace nlpp;


int main ()
{
    using Func = Rosenbrock;
    using IH = BFGS_Diagonal;
    using LS = DynamicLineSearch<Func>;
    using Out = out::GradientOptimizer<0>;

    params::LBFGS<IH, LS, Out> params;

    params.maxIterations = 1e4;
    params.fTol = 1e-6;
    params.gTol = 1e-6;
    params.xTol = 1e-6;

    LBFGS<Vec, IH, LS, Out> lbfgs(params);

    Func func;
    Vec x = Vec::Constant(500, 1.2);

    handy::benchmark([&]
    {
        x = lbfgs(func, x);
    });

    handy::print(x.transpose());


    return 0;
}