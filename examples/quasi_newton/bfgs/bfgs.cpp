#include "quasi_newton/bfgs/bfgs.hpp"
#include "TestFunctions/Rosenbrock.h"


int main ()
{
    using nlpp::wrap::Conditions;
    using V = Eigen::Vector4d;
    // using V = nlpp::Vec;

    using Opt = nlpp::BFGS<>;

    Opt opt;
    nlpp::Rosenbrock func;
    V x0 = V::Constant(4, 2.0);


    handy::print(handy::benchmark([&]{
        x0 = opt(func, x0, Opt::constraints());
    }));

    handy::print(x0.transpose());

    return 0;
}

