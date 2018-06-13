#include "QuasiNewton/LBFGS/LBFGS.h"

#include "TestFunctions/Rosenbrock.h"


using namespace nlpp;


int main ()
{
    using IH = BFGS_Diagonal;
    using LS = StrongWolfe;
    using Out = out::GradientOptimizer<0>;

    params::LBFGS<IH, LS, Out> params;

    params.maxIterations = 1e4;
    params.fTol = 1e-6;
    params.gTol = 1e-6;
    params.xTol = 1e-6;

    LBFGS<Vec, IH, LS, Out> lbfgs(params);

    Vec x = Vec::Constant(500, 1.2);

    handy::benchmark([&]
    {
        x = lbfgs(Rosenbrock{}, x);
    });

    handy::print(x.transpose());


    return 0;
}