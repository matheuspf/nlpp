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

    nlpp::CG<> opt;
    nlpp::Rosenbrock func;
    // V x0 = V::Constant(4, 2.0);
    V x0 = V::Constant(2.0);


    handy::print(handy::benchmark([&]{
        x0 = opt.opt(nlpp::wrap::functions<Conditions::Function | Conditions::Gradient>(func, nlpp::fd::gradient(func)),
                    nlpp::wrap::domain<Conditions::Start>(x0),
                    nlpp::wrap::constraints<Conditions::Empty>());
    }));

    handy::print(x0.transpose());

    return 0;
}
