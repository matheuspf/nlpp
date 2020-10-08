#include <gtest/gtest.h>

#include "utils/stop.hpp"


namespace
{

TEST(StopTest, SimpleTest)
{
    using namespace nlpp;

    Vec x = Vec::Constant(3, 2.0);
    Vec gx = Vec::Constant(3, 4.0);
    Vec::Scalar fx = 1.0;

    using Float = types::Float;

    int maxIterations = 1e4;
    Float xTol = 1e-4;
    Float fTol = 1e-4;
    Float gTol = 1e-4;

    stop::Optimizer<false, double> opt0(maxIterations, xTol, fTol);
    stop::Optimizer<true, double> opt1(maxIterations, xTol, fTol);

    EXPECT_EQ(opt0(nullptr, x, fx), Status::Code::Ok);
    EXPECT_EQ(opt0(nullptr, x, fx), Status::Code::VariableCondition | Status::Code::FunctionCondition);
    EXPECT_EQ(opt0(nullptr, x, fx + 2*fTol), Status::Code::VariableCondition);

    EXPECT_EQ(opt1(nullptr, x, fx), Status::Code::Ok);
    EXPECT_EQ(opt1(nullptr, x, fx), Status::Code::VariableCondition | Status::Code::FunctionCondition);
    EXPECT_EQ(opt1(nullptr, x, fx + 2*fTol), Status::Code::Ok);


    stop::GradientOptimizer<false, double> gopt0(maxIterations, xTol, fTol, gTol);
    stop::GradientOptimizer<true, double> gopt1(maxIterations, xTol, fTol, gTol);

    EXPECT_EQ(gopt0(nullptr, x, fx, gx), Status::Code::Ok);
    EXPECT_EQ(gopt0(nullptr, x, fx, gx), Status::Code::VariableCondition | Status::Code::FunctionCondition);
    EXPECT_EQ(gopt0(nullptr, x, fx + 2*fTol, gx / 1e5), Status::Code::VariableCondition | Status::Code::GradientCondition);

    EXPECT_EQ(gopt1(nullptr, x, fx, gx), Status::Code::Ok);
    EXPECT_EQ(gopt1(nullptr, x, fx, gx / 1e5), Status::Code::VariableCondition | Status::Code::FunctionCondition | Status::Code::GradientCondition);
    EXPECT_EQ(gopt1(nullptr, x, fx + 2*fTol, gx / 1e5), Status::Code::Ok);


    stop::GradientNorm gnopt(maxIterations, gTol);

    EXPECT_EQ(gnopt(nullptr, x, fx, gx), Status::Code::Ok);
    EXPECT_EQ(gnopt(nullptr, x, fx, Vec::Constant(3, gTol)), Status::Code::GradientCondition);
}

// 


} // namespace