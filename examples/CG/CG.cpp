#include "CG/CG.h"

#include "TestFunctions/Rosenbrock.h"

// #include "LineSearch/Goldstein/Goldstein.h"

using namespace nlpp;



int main ()
{
    // using LS = StrongWolfe<>;
    // using Stop = stop::GradientOptimizer<true>;
    // using Out = out::GradientOptimizer<>;

    // CG<FR_PR, LS, Stop, Out> opt;
    ::nlpp::poly::CG<> opt;
    
    // opt.stop.maxIterations_ = 1e5;
    // opt.stop.fTol = 1e-8;
    // opt.stop.fTol = 1e-8;
    // opt.stop.fTol = 1e-8;

    Vec x = Vec::Constant(5, 2.0);

    Rosenbrock func;

    auto res = opt(func, x);

    handy::print(res.transpose());



    return 0;
}