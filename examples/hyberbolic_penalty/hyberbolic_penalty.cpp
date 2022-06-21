#include "hyperbolic_penalty/hyperbolic_penalty.hpp"
#include "TestFunctions/TreeBarTruss.h"
#include "TestFunctions/hock_schittkowski.hpp"

#include "cg/cg.hpp"


int main ()
{
    using nlpp::wrap::Conditions;
    // using V = Eigen::Vector4d;
    using V = nlpp::Vec;

    using Opt = nlpp::HyperbolicPenalty<>;


    Opt opt;
    opt.stop.maxIterations = 10;

    nlpp::TreeBarTrussFunc func;
    nlpp::TreeBarTrussIneqs constraints;

    // nlpp::P95<> func;
    // nlpp::P95<> constraints;

    V x = V::Constant(2, 0.5);

    handy::print(handy::benchmark([&]{
        x = opt(func, x, constraints);
        // x = opt(func, x, nlpp::wrap::constraints(constraints, {}, {}));
    }));

    handy::print(x.transpose(), "\t", func(x));


    return 0;
}