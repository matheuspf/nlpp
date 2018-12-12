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

	using CGType = nlpp::FR_PR;
	using LS = nlpp::StrongWolfe<>;
	using Stop = nlpp::stop::GradientOptimizer<1>;
	using Out = nlpp::out::GradientOptimizer<2>;

};



TEST_F(CG, PerformanceTest)
{
	nlpp::params::CG<CGType, LS, Stop, Out> params;

	params.stop.fTol = 1e-3;
	params.stop.xTol = 1e-3;
	params.stop.gTol = 1e-3;
	params.stop.maxIterations = 1e4;

	nlpp::CG<CGType, LS, Stop, Out> cg(params);

	nlpp::Rosenbrock func;
	auto grad = nlpp::fd::gradient(func);

	nlpp::Vec x0 = nlpp::Vec::Constant(10, 1.2);
		
	nlpp::Vec x = cg(func, grad, x0);


	EXPECT_LE(grad(x).norm(), 1e-3);

	EXPECT_LE(cg.output.vX.size(), params.stop.maxIterations + 1);
}




} // namespace

