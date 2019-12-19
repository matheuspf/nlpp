#include "cg/cg.hpp"
#include "TestFunctions/Rosenbrock.h"


int main ()
{
    nlpp::CG<> opt;

    nlpp::Rosenbrock func;
    nlpp::Vec x0 = nlpp::Vec::Constant(10, 2.0);

    // auto f = nlpp::wrap::makeFuncGrad<nlpp::Vec>(func);

    // using F = decltype(f);
    // using F = nlpp::wrap::FunctionGradient<nlpp::Rosenbrock>;
    // using F = nlpp::wrap::Function<nlpp::wrap::Gradient<nlpp::Rosenbrock>>;
    // using V = nlpp::Vec;

    // handy::print(nlpp::wrap::impl::isFunction<F, V>, nlpp::wrap::impl::isGradient_1<F, V>, nlpp::wrap::impl::isFuncGrad_2<F, V>);
    // handy::print(f.function(x0));

    auto res = opt(func, x0);

    // handy::print(res.transpose());


    return 0;
}
