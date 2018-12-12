#include "gtest/gtest.h"

#include "Newton/Newton.h"

#include "TestFunctions/Rosenbrock.h"



namespace
{

// struct Newton : public ::testing::Test
// {
//     virtual ~Newton ()
//     {
//     }

//     virtual void SetUp () {}

//     virtual void TearDown () {}
// };


TEST(Newton, PrecisionTest)
{
    using Opt = nlpp::Newton<nlpp::fact::CholeskyIdentity<>, nlpp::StrongWolfe<>,
                            nlpp::stop::GradientOptimizer<1>, nlpp::out::GradientOptimizer<0>>;


    typename Opt::Params params;

    params.stop.xTol = 1e-4;
    params.stop.fTol = 1e-4;
    params.stop.gTol = 1e-4;


    int N = 10;

    double x0 = 1.2;


    nlpp::Rosenbrock func;

    auto grad = nlpp::fd::gradient(func);

    auto hess = nlpp::fd::hessian(func);


    nlpp::Vec xOpt = nlpp::Vec::Constant(N, 1.0);

    double fOpt = 0.0;


    Opt newton(params);

    nlpp::Vec x = nlpp::Vec::Constant(N, x0);


	x = newton(func, grad, x, hess);


    ASSERT_LE((x - xOpt).norm(), params.stop.xTol);

    ASSERT_LE(std::abs(func(x) - fOpt), params.stop.fTol);
    
    ASSERT_LE(grad(x).norm(), params.stop.gTol);
}





} // namespace