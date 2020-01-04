#include "QuasiNewton/BFGS/BFGS.h"
#include "QuasiNewton/LBFGS/LBFGS.h"

#include "LineSearch/Goldstein/Goldstein.hpp"

#include "TestFunctions/Rosenbrock.h"

#include "Newton/Newton.h"


using namespace nlpp;


int main ()
{
    //BFGS<BFGS_Diagonal<>, StrongWolfe<>, stop::GradientOptimizer<>, out::BFGS<>> bfgs;
    // poly::BFGS<> bfgs;
    poly::LBFGS<> bfgs;
    //poly::Newton<fact::CholeskyIdentity<>> bfgs;

    // bfgs.output = std::make_unique<out::poly::BFGS<>>();
    bfgs.stop = std::make_unique<stop::poly::GradientOptimizer<>>(10000, 1e-4, 1e-4, 1e-4);

    Rosenbrock func;
    auto grad = fd::gradient(func);
    Vec x0(1000);

    std::for_each(x0.data(), x0.data() + x0.size(), [](auto& xi){ xi = handy::rand(-10.0, 10.0); });


    auto x = bfgs(func, grad, x0);

    handy::print(x.transpose());

    // std::cout << "\n\n" << fd::hessian(Rosenbrock{})(x).inverse() << "\n\n";


    return 0;
}

// int main ()
// {
//     using BFGS = BFGS<BFGS_Diagonal<>, StrongWolfe<>, stop::GradientOptimizer<1>, out::GradientOptimizer<1>>;

//     BFGS::Params params;

//     params.stop.maxIterations = 1e1;
//     params.stop.gTol = 1e-4;
//     params.stop.fTol = 1e-4;
//     params.stop.xTol = 1e-4;
//     params.lineSearch = StrongWolfe<>(1.0, 1e-4, 0.9);


//     BFGS bfgs(params);

//     Rosenbrock func;

//     VecX<double> x = VecX<double>::Constant(10, 1.2);
//     //VecX<float> x = VecX<float>::Constant(10, 1.2);

//     //Vec x(2); x << -1.2, 1;


//     handy::benchmark([&]
//     {
//         x = bfgs(func, x);
//     });

//     handy::print(x.transpose());


//     return 0;
// }