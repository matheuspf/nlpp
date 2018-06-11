#include "gtest/gtest.h"

#include "GradientDescent/GradientDescent.h"
#include "TestFunctions/Rosenbrock.h"


namespace
{

struct GradientDescent : public ::testing::Test
{
	GradientDescent () { }

	virtual ~GradientDescent () { }


	virtual void SetUp ()
	{
	}

	virtual void TearDown ()
	{
	}


	nlpp::GradientDescent<nlpp::Goldstein> cg;
};



TEST_F(GradientDescent, PerformanceTest)
{
	nlpp::params::GradientDescent<nlpp::Goldstein> params;

	params.fTol = 0.0;
	params.xTol = 0.0;
	params.gTol = 1e-3;
	params.maxIterations = 1e4;

	cg = nlpp::GradientDescent<nlpp::Goldstein>(params);

	nlpp::Rosenbrock func;
	auto grad = nlpp::fd::gradient(func);

	nlpp::Vec x0 = nlpp::Vec::Constant(10, 1.2);
		
	nlpp::Vec x = cg(func, grad, x0);

	EXPECT_LE(grad(x).norm(), 1e-3);
}




} // namespace

