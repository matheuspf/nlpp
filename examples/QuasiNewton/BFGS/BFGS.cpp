#include "QuasiNewton/BFGS/BFGS.h"

#include "LineSearch/Goldstein/Goldstein.h"

#include "TestFunctions/Rosenbrock.h"


using namespace nlpp;


int main ()
{
    using BFGS = BFGS<BFGS_Diagonal<>, StrongWolfe, stop::GradientOptimizer<1>, out::GradientOptimizer<1>>;

    BFGS::Params params;

    params.stop.maxIterations = 1e1;
    params.stop.gTol = 1e-4;
    params.stop.fTol = 1e-4;
    params.stop.xTol = 1e-4;
    params.lineSearch = StrongWolfe(1.0, 1e-4, 0.9);


    BFGS bfgs(params);

    Rosenbrock func;

    VecX<double> x = VecX<double>::Constant(10, 1.2);
    //VecX<float> x = VecX<float>::Constant(10, 1.2);

    //Vec x(2); x << -1.2, 1;


    handy::benchmark([&]
    {
        x = bfgs(func, x);
    });

    handy::print(x.transpose());


    return 0;
}