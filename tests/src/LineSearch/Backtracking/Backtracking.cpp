#include "gtest/gtest.h"

#include "LineSearch/Backtracking/Backtracking.h"

#include "GradientDescent/GradientDescent.h"


#include "TestFunctions/Rosenbrock.h"


namespace
{

struct Backtracking : ::testing::Test
{
    virtual ~Backtracking ()
    {
    }

    virtual void SetUp ()
    {
    }

    virtual void TearDown ()
    {
    }


    nlpp::GradientDescent<nlpp::Backtracking> gd;
};


TEST_F(Backtracking, BacktrackingTest)
{
	handy::Benchmark benchmark;

	nlpp::params::GradientDescent<nlpp::Backtracking> params;

	params.fTol = 1e-3;
	params.xTol = 1e-3;
	params.gTol = 1e-3;
	params.maxIterations = 1e4;

	gd = nlpp::GradientDescent<nlpp::Backtracking>(params);

	nlpp::Rosenbrock func;
	auto grad = nlpp::fd::gradient(func);

	nlpp::Vec x0 = nlpp::Vec::Constant(10, 1.2);
	
	nlpp::Vec x = gd(func, grad, x0);

	EXPECT_LE(grad(x).norm(), 1e-3);

	EXPECT_LE(benchmark.finish(), 1e-1);
}


} // namespace