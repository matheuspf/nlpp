#include "lib/cpp/include/gradient_descent/gradient_descent.hpp"
#include "TestFunctions/Rosenbrock.h"


int main ()
{
    nlpp::poly::GradientDescent<> opt_;
    nlpp::poly::LineSearchOptimizer<>& opt = opt_;

    opt.stop = nlpp::poly::stop::GradientOptimizer<true>(1e3);
    opt.lineSearch = nlpp::poly::Goldstein<>();

    nlpp::Rosenbrock func;
    nlpp::Vec x0 = nlpp::Vec::Constant(5, 2.0);

    auto res = opt(func, x0);

    handy::print(res.transpose());


    return 0;
}