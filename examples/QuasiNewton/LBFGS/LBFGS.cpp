#include "QuasiNewton/LBFGS/LBFGS.h"

#include "LineSearch/Dynamic/Dynamic.h"

#include "LineSearch/StrongWolfe/StrongWolfe.h"

#include "LineSearch/Goldstein/Goldstein.h"

#include "TestFunctions/Rosenbrock.h"


using namespace nlpp;


int main ()
{
    using Func = Rosenbrock;
    using IH = BFGS_Diagonal<>;
    using LS = StrongWolfe;
    using Stop = stop::GradientOptimizer<0>;
    using Out = out::GradientOptimizer<1>;

    params::LBFGS<IH, LS, Stop, Out> params;

    params.stop.maxIterations = 1e4;
    params.stop.fTol = 1e-4;
    params.stop.gTol = 1e-4;
    params.stop.xTol = 1e-4;
    params.m = 10;

    LBFGS<IH, LS, Stop, Out> lbfgs(params);

    Func func;
    Vec x = Vec::Constant(50, 1.2);

    handy::benchmark([&]
    {
        x = lbfgs(func, x);
    });

    handy::print(x.transpose());


    return 0;
}