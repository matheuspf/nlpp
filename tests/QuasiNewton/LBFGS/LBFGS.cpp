#include "gtest/gtest.h"

#include "QuasiNewton/LBFGS/LBFGS.h"

#include "TestFunctions/Rosenbrock.h"


namespace 
{

struct LBFGS_Test : public ::testing::Test
{
    virtual ~LBFGS_Test ()
    {
    }

    virtual void SetUp ()
    {
    }

    virtual void TearDown ()
    {
    }


    using Func = nlpp::Rosenbrock;
    using IH = nlpp::BFGS_Diagonal<>;
    using LS = nlpp::StrongWolfe<>;
    using Stop = nlpp::stop::GradientOptimizer<1>;
    using Out = nlpp::out::GradientOptimizer<0>;
};


TEST_F(LBFGS_Test, LBFGS_PerformanceTest)
{
    Func func;

    auto grad = nlpp::fd::gradient(func);

    nlpp::params::LBFGS<IH, LS, Stop, Out> params;

    params.stop.maxIterations = 1e4;
    params.stop.fTol = 1e-3;
    params.stop.gTol = 1e-3;
    params.stop.xTol = 1e-3;
    params.m = 10;

    nlpp::LBFGS<IH, LS, Stop, Out> lbfgs(params);

    nlpp::Vec x = nlpp::Vec::Constant(10, 1.2);

    double tm = handy::benchmark([&]
    {
        x = lbfgs(func, grad, x);
    });


    EXPECT_LE(tm, 0.1);

    EXPECT_LE(grad(x).norm(), 1e-3);
}


}