#include "cg/cg.hpp"
#include "TestFunctions/Rosenbrock.h"


template <class Impl, class Func, class V>
void exec (const nlpp::LineSearchOptimizer<Impl>& opt, const Func& func, const Eigen::MatrixBase<V>& x0)
{
    nlpp::Vec res;
    
    handy::print(handy::benchmark([&]{
        res = opt(func, x0);
    }));

    handy::print(res.transpose());

}



int main ()
{
    nlpp::CG<> opt;
    nlpp::Rosenbrock func;
    nlpp::Vec x0 = nlpp::Vec::Constant(10, 2.0);

    exec(opt, func, x0);

    return 0;
}
