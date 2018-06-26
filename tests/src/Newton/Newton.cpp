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
    using Opt = nlpp::Newton<nlpp::fact::CholeskyIdentity, nlpp::StrongWolfe,
                            nlpp::stop::GradientOptimizer<1>, nlpp::out::GradientOptimizer<1>>;


    typename Opt::Params params;

    params.stop.xTol = 1e-4;
    params.stop.fTol = 1e-4;
    params.stop.gTol = 1e-4;


    int N = 50;

    double x0 = 5.0;


    nlpp::Rosenbrock func;

    auto grad = nlpp::fd::gradient(func);

    auto hess = nlpp::fd::hessian(func);


    nlpp::Vec xOpt = nlpp::Vec::Constant(N, x0);

    double fOpt = 0.0;


    Opt newton(params);

    nlpp::Vec x = nlpp::Vec::Constant(N, x0);


	x = newton(func, nlpp::fd::gradient(func), x, nlpp::fd::hessian(func));


    ASSERT_LE((x - xOpt).norm(), params.stop.xTol);

    ASSERT_LE(std::abs(func(x)), params.stop.fTol);
    
    ASSERT_LE(grad(x).norm(), params.stop.gTol);
}





} // namespace