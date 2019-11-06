#include <nlpp/nlpp.hpp>


void test1 ()
{
    nlpp::CG cg(1e-4, nlpp::ls::StrongWolfe(1e-4), nlpp::stop::GradientOptimizer(2), nlpp::out::GradientOptimizer(2));
    cg.ls = nlpp::ls::Goldstein(1e-4)

    auto func = [](const auto& x){ return (x - 0.5).norm(); };

    nlpp::Vec x0(2);
    x0 << 2.0, 0.0;

    auto res = cg(func, x0);

    handy::print(res.transpose());
}

void test2 ()
{
    std::unique_ptr<nlpp::GradientOptimizer> cg = std::make_unique<nlpp::CG>(1e-4, nlpp::ls::StrongWolfe(1e-4));
    opt->stop = nlpp::stop::GradientOptimizer(2);
    opt->out = nlpp::out::GradientOptimizer(2);

    nlpp::Vec x0(2);
    x0 << 2.0, 0.0;

    auto res = (*cg)(func, x0);

    handy::print(res.transpose());
}


int main ()
{
    test1();
    test2();

    return 0;
}