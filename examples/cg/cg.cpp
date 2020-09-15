#include "cg/cg.hpp"
#include "TestFunctions/Rosenbrock.h"


// template <class Impl, class Func, class V>
// void exec (nlpp::LineSearchOptimizer<Impl>& opt, const Func& func, const Eigen::MatrixBase<V>& x0)
// {
//     nlpp::Vec res;
    
//     handy::print(handy::benchmark([&]{
//         res = opt(func, x0);
//     }));

//     handy::print(res.transpose());
// }


int main ()
{
    using nlpp::wrap::Conditions;
    using V = Eigen::Vector4d;
    // using V = nlpp::Vec;
    using Opt = nlpp::CG<>;

    Opt opt;
    nlpp::Rosenbrock func;
    V x0 = V::Constant(4, 2.0);

    handy::Benchmark bench;

    auto [x, fx, gx, status] = opt.opt(Opt::functions<V>(func, nlpp::fd::gradient(func)),
                                       Opt::domain(x0),
                                       Opt::constraints());
    // x0 = opt(func, x0, Opt::constraints());

    handy::print(bench.finish());

    handy::print(x.transpose());

    return 0;
}
