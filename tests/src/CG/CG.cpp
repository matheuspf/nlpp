#include "gtest/gtest.h"

#include "CG/CG.h"
#include "TestFunctions/Rosenbrock.h"


namespace
{

struct CG : public ::testing::Test
{
	CG () { }

	virtual ~CG () { }


	virtual void SetUp ()
	{
	}

	virtual void TearDown ()
	{
	}


	nlpp::CG<> cg;
};



TEST_F(CG, PerformanceTest)
{
	nlpp::params::CG<> params;

	params.stop.fTol = 0.0;
	params.stop.xTol = 0.0;
	params.stop.gTol = 1e-3;
	params.stop.maxIterations = 1e4;

	cg = nlpp::CG<>(params);

	nlpp::Rosenbrock func;
	auto grad = nlpp::fd::gradient(func);

	nlpp::Vec x0 = nlpp::Vec::Constant(10, 1.2);
		
	nlpp::Vec x = cg(func, grad, x0);

	EXPECT_LE(grad(x).norm(), 1e-3);
}




} // namespace

