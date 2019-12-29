#include "cg/cg.hpp"
#include "TestFunctions/Rosenbrock.h"


struct F1
{
    double operator() (const nlpp::Vec& x, nlpp::Vec& gx) const
    {
        // if(calcGrad)
            gx = g(x);

        return f(x);
    }

    std::pair<double, nlpp::Vec> funcGrad (const nlpp::Vec& x) const
    {
        return {f(x), g(x)};
    }

    auto function (const nlpp::Vec& x) const
    {
        return f(x);
    }

    void gradient (const nlpp::Vec& x, nlpp::Vec& gx) const
    {
        g(x, gx);
    }

    nlpp::Rosenbrock f;
    nlpp::fd::Gradient<nlpp::Rosenbrock> g = nlpp::fd::Gradient<nlpp::Rosenbrock>(f);
};

int main ()
{
    nlpp::CG<nlpp::FR_PR, nlpp::StrongWolfe<>, nlpp::stop::GradientOptimizer<true>> opt;

    nlpp::Rosenbrock func;
    nlpp::Vec x0 = nlpp::Vec::Constant(10, 2.0);

    auto f1 = [ff=nlpp::wrap::functionGradient(func, nlpp::fd::gradient(func))](const nlpp::Vec& x, nlpp::Vec& g, bool calcGrad) -> double { return ff(x, g, calcGrad); };
    // auto f1 = [ff=nlpp::wrap::functionGradient(func, nlpp::fd::gradient(func))](const nlpp::Vec& x) { return ff(x); };
    auto f2 = nlpp::wrap::functionGradient(f1);

    // handy::print(decltype(f2)::HasOp<nlpp::Vec>::FuncGrad);

    // auto r = f2(x0);

    // auto res = opt(f2, x0);

    nlpp::Vec res;

    handy::print(handy::benchmark([&]{
        res = opt(F1{}, x0);
    }));

    handy::print(res);


    return 0;
}
