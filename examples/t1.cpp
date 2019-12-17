#include "cg/cg.hpp"
#include "TestFunctions/Rosenbrock.h"


int main ()
{
    nlpp::CG<> opt;

    nlpp::Rosenbrock func;
    nlpp::Vec x0 = nlpp::Vec::Constant(10, 2.0);

    auto res = opt(func, x0);

    handy::print(res.transpose());


    return 0;
}
