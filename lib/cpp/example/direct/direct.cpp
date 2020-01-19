#include "lib/cpp/include/direct/direct.hpp"


int main ()
{
    nlpp::Direct<> opt;

    auto func = [](const nlpp::Vec& x) -> double { return x.dot(x); };

    nlpp::Vec l = nlpp::Vec::Constant(5, -1.0);
    nlpp::Vec u = nlpp::Vec::Constant(5, 1.0);

    nlpp::Vec res = opt.optimize(nlpp::wrap::poly::Builder<nlpp::Vec>::function(func), l, u);

    handy::print(res.transpose());

    return 0;
}
