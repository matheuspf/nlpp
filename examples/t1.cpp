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

struct F2
{
    template <class V>
    auto funcGrad (const V& x, V& g) const
    {
        auto ff = nlpp::wrap::functionGradient(f, nlpp::fd::gradient(f));

        return ff(x, g);
    }

    nlpp::Rosenbrock f;
};


template <class V>
void test ()
{
    nlpp::Rosenbrock func;
    auto f1 = [ff=nlpp::wrap::functionGradient(func, nlpp::fd::gradient(func))](const auto& x, auto& g){ return ff.funcGrad(x, g); };
    // auto rrr = nlpp::wrap::impl::gradientType<decltype(f1), const Eigen::MatrixBase<V>&, const Eigen::MatrixBase<V>&>{};
    // auto rrr = nlpp::wrap::impl::gradientType<decltype(f1), const Eigen::MatrixBase<V>&, nlpp::impl::Plain<V>&>{};
    // auto rrr = nlpp::wrap::impl::gradientType<decltype(f1), const V&, const V&>{};

    // nlpp::impl::PrintType<decltype(rrr)>{};

    // handy::print(nlpp::impl::is_detected_v<std::invoke_result_t, decltype(f1), const V&, V&>);
    handy::print(nlpp::impl::is_detected_v<std::invoke_result_t, decltype(f1), nlpp::impl::Plain<V>, nlpp::impl::Plain<V>>);



    exit(0);
}

int main ()
{
    // test<nlpp::Vec>();
    // test<decltype(nlpp::Vec() + nlpp::Vec())>();


    nlpp::CG<nlpp::FR_PR, nlpp::StrongWolfe<>, nlpp::stop::GradientOptimizer<true>> opt;

    nlpp::Rosenbrock func;
    nlpp::Vec x0 = nlpp::Vec::Constant(10, 2.0);

    // auto f1 = [ff=nlpp::wrap::functionGradient(func, nlpp::fd::gradient(func))](const auto& x, auto& g, bool calcGrad){ return ff.funcGrad(x, g, calcGrad); };
    auto f1 = [ff=nlpp::wrap::functionGradient(func, nlpp::fd::gradient(func))](const auto& x, auto& g){ return ff.funcGrad(x, g); };
    // auto f1 = [ff=nlpp::wrap::functionGradient(func, nlpp::fd::gradient(func))](const nlpp::Vec& x) { return ff(x); };
    auto f2 = nlpp::wrap::functionGradient(f1);

    // handy::print(nlpp::wrap::impl::IsDirectional<decltype(f1), nlpp::Vec>::value);
    // handy::print(nlpp::impl::is_detected_v<std::invoke_result_t, decltype(f1), nlpp::Vec>);
    // handy::print(nlpp::impl::is_detected_v<std::invoke_result_t, F2, nlpp::Vec>);


    // handy::print(decltype(f2)::HasOp<nlpp::Vec>::FuncGrad);

    // nlpp::Vec g(x0.rows());
    // auto r = f2.funcGrad(x0, g);
    // handy::print(r);

    // auto res = opt(f2, x0);

    nlpp::Vec res;

    handy::print(handy::benchmark([&]{
        res = opt(f1, x0);
    }));

    handy::print(res);


    return 0;
}
