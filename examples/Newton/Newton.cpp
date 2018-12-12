#include "Newton/Newton.h"

#include "TestFunctions/Rosenbrock.h"

using namespace nlpp;


int main ()
{
    using F = long double;

    using Opt = nlpp::Newton<nlpp::fact::CholeskyIdentity<>, nlpp::StrongWolfe<>,
                            nlpp::stop::GradientOptimizer<1>, nlpp::out::GradientOptimizer<>>;
                            

    typename Opt::Params params;

    params.stop.xTol = 1e-4;
    params.stop.fTol = 1e-4;
    params.stop.gTol = 1e-4;


    int N = 10;

    F x0 = 1.2;


    nlpp::Rosenbrock func;

    auto grad = fd::gradient(func);
    auto hess = fd::hessian(func);

    Opt newton(params);

    nlpp::VecX<F> x = VecX<F>::Constant(N, x0);


	x = newton(func, grad, x, hess);


    handy::print("x: ", x.transpose(), "\nfx: ", func(x), "\ngx: ", grad(x).norm(), "\n");





	return 0;
}
