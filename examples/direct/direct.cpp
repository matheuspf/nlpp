#include "direct/direct.hpp"


template <class Impl>
void foo (const nlpp::BoundConstrainedOptimizer<Impl>& opt)
{
    auto func = [](const nlpp::Vec& x) -> double { return x.dot(x); };

    nlpp::Vec l = nlpp::Vec::Constant(5, -1.0);
    nlpp::Vec u = nlpp::Vec::Constant(5, 1.0);

    nlpp::Vec res = opt(func, l, u);

    handy::print(res.transpose());
}


int main ()
{
    nlpp::Direct<> opt;
    foo(opt);


    return 0;
}
