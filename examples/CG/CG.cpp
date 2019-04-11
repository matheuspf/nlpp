#include "CG/CG.h"

#include "TestFunctions/Rosenbrock.h"

#include "LineSearch/Goldstein/Goldstein.h"

// #include "Newton/Newton.h"

using namespace nlpp;



int main ()
{
    poly::CG<> opt;

    opt.output = out::poly::GradientOptimizer<1>();
    opt.stop = stop::poly::GradientOptimizer<true>(10000, 1e-1, 1e-1, 1e-1);

    Vec x = Vec::Constant(5, 2.0);

    auto res = opt(Rosenbrock{}, x);

    handy::print(res.transpose());


    // Newton<fact::CholeskyIdentity<double>, StrongWolfe<double, FirstOrderStep<double>>, stop::GradientOptimizer<true>, out::GradientOptimizer<0>> opt;

    // opt.stop = stop::GradientOptimizer<true>(10000, 1e-4, 1e-4, 1e-4);

	// int count = 0;

	// auto func = [&]{ Rosenbrock func; return [&](const auto& x){ count++; return func(x); }; }();

    // auto grad = fd::gradient(func);

    // Vec x0(100);

    // std::for_each(x0.data(), x0.data() + x0.size(), [](auto& xi){ xi = handy::rand(-10.0, 10.0); });

    // auto x = opt(func, grad, x0);

    // handy::print(x.transpose(), "\n\n");
	// handy::print(count);

    return 0;
}
