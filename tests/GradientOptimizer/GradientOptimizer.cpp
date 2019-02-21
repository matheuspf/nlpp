#include "gtest/gtest.h"

#include "GradientDescent/GradientDescent.h"
#include "CG/CG.h"

#include "TestFunctions/Rosenbrock.h"


namespace
{

struct GradientOptimizerTest : public ::testing::Test
{
    virtual ~GradientOptimizerTest () {}
    
    template <class Optimizer, class Function, class Vec>
    void convergenceTest (Optimizer optimizer, Function func, Vec x0)
    {
        auto grad = ::nlpp::fd::gradient(func);

        auto x = optimizer(func, grad, x0);
        
        auto fx = func(x);
        auto gx = grad(x);

        auto testCallback = [&]
        {
            if(optimizer.stop(optimizer, x, fx, gx))
                return ::testing::AssertionSuccess();
            
            else
                return ::testing::AssertionFailure() << "Failed with:\tfx: " << (fx*fx) << "\tgx: " << gx.dot(gx);
        };

        ASSERT_TRUE(testCallback());
    }
};


TEST_F(GradientOptimizerTest, CG)
{
    SCOPED_TRACE("CG Test");

    ::nlpp::poly::CG<> opt;

    opt.stop = new ::nlpp::stop::poly::GradientOptimizer<>(10000, 1e-4, 1e-5, 1e-5);
    
    ::nlpp::Rosenbrock func;

    for(int numVariables = 10; numVariables <= 100; numVariables += 10)
    {
        SCOPED_TRACE((std::string("Rosenbrock \t N: ") + std::to_string(numVariables)).c_str());

        convergenceTest(opt, func, ::nlpp::Vec::Constant(numVariables, -5.0));
    }
}


} // namespace