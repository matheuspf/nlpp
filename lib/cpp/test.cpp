#include "lib/cpp/include/cg/cg.hpp"
#include "TestFunctions/Rosenbrock.h"


int main ()
{
    nlpp::poly::CG<> opt;
    // Opt<CGType, LS, Stop, Out> opt(LS{}, 1e4);

    nlpp::Rosenbrock func;
    nlpp::Vec x0 = nlpp::Vec::Constant(10, 2.0);

    nlpp::Vec res;
    
    handy::print(handy::benchmark([&]{
        res = opt(func, x0);
    }));

    handy::print(res.transpose());


    return 0;
}
