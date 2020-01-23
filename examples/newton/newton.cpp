#include "newton/newton.hpp"
#include "TestFunctions/Rosenbrock.h"


int main ()
{
    // using Opt = nlpp::Newton<nlpp::fact::CholeskyIdentity<>, nlpp::StrongWolfe<>,
    //                         nlpp::stop::GradientOptimizer<1>, nlpp::out::GradientOptimizer<>>;

    using Opt = nlpp::poly::Newton<nlpp::fact::CholeskyIdentity<>>;

                            

    // impl::params::Newton<params::LineSearchOptimizer<nlpp::StrongWolfe<>,
    //                         nlpp::stop::GradientOptimizer<1>, nlpp::out::GradientOptimizer<>>, nlpp::fact::CholeskyIdentity<>> params;

    // params.stop. xTol = 1e-4;
    // params.stop.fTol = 1e-4;
    // params.stop.gTol = 1e-4;

    impl::params::Newton<params::poly::LineSearchOptimizer_, nlpp::fact::CholeskyIdentity<>> params;

    params.stop = std::make_unique<stop::poly::GradientOptimizer<>>(1e-4, 1e-4, 1e-4);


    int N = 10;

    double x0 = 5.0;


    nlpp::Rosenbrock func;

    auto grad = fd::gradient(func);
    auto hess = fd::hessian(func);

    // Opt newton(params);
    Opt newton;

    newton.init();

    Vec x = Vec::Constant(N, x0);

	// x = newton(func, grad, hess, x);
	x = newton(func, grad, x);


    handy::print("x: ", x.transpose(), "\nfx: ", func(x), "\ngx: ", grad(x).norm(), "\n");

	return 0;
}
