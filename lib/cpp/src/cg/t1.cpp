#include "lib/cpp/include/cg/cg.hpp"
#include "TestFunctions/Rosenbrock.h"


int main ()
{
    nlpp_p::CG<> opt;
    // Opt<CGType, LS, Stop, Out> opt(LS{}, 1e4);

    // opt.cg = nlpp_p::HS{};
    opt.stop = nlpp::poly::stop::GradientOptimizer<true>(1e3);

    nlpp::Rosenbrock func;
    nlpp::Vec x0 = nlpp::Vec::Constant(20, 20.0); x0[3] = -20.0;

    auto res = opt(func, x0);

    handy::print(res.transpose());


    return 0;
}
