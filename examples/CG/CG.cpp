#include "CG/CG.h"

#include "TestFunctions/Rosenbrock.h"

#include "LineSearch/Goldstein/Goldstein.h"

// #include "Newton/Newton.h"

using namespace nlpp;



int main ()
{
    CG<> opt;

    // opt.output = out::poly::GradientOptimizer<1>();
    // opt.stop = stop::poly::GradientOptimizer<true>(10000, 1e-1, 1e-1, 1e-1);

    // Vec x = Vec::Constant(5, 2.0);

    // Rosenbrock func;

    // auto res = opt(func, x);

    // handy::print(res.transpose());


    return 0;
}