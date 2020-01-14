#include "lib/cpp/include/cg/cg.hpp"
#include "TestFunctions/Rosenbrock.h"


int main ()
{
    nlpp::poly::CG<> opt_;
    nlpp::poly::GradientOptimizer<>& opt = opt_;
    // Opt<CGType, LS, Stop, Out> opt(LS{}, 1e4);

    // opt.cg = nlpp::poly::HS{};
    opt.stop = nlpp::poly::stop::GradientOptimizer<true>(1e3);

    nlpp::Rosenbrock func;
    nlpp::Vec x0 = nlpp::Vec::Constant(20, 20.0); x0[3] = -20.0;

    auto res = opt(func, x0);

    handy::print(res.transpose());


    return 0;
}
