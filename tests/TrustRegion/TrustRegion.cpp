#include <gtest/gtest.h>

#include "TrustRegion/CauchyPoint/CauchyPoint.h"
#include "TrustRegion/DogLeg/DogLeg.h"
#include "TrustRegion/IndefiniteDogLeg/IndefiniteDogLeg.h"
#include "TrustRegion/IterativeTR/IterativeTR.h"

#include "TestFunctions/Rosenbrock.h"


namespace
{

struct TrustRegionTest : public ::testing::Test
{
    virtual ~TrustRegionTest () {}
    
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
                return ::testing::AssertionFailure() << "Failed with:\tfx: " << fx << "\tgx: " << gx.norm();
        };

        ASSERT_TRUE(testCallback());
    }
};


TEST_F(TrustRegionTest, CauchyPoint)
{
    SCOPED_TRACE("Cauchy Point Test");

    ::nlpp::CauchyPoint<::nlpp::stop::GradientNorm<>> opt;
    opt.stop = ::nlpp::stop::GradientNorm<>(10000, 1e-3);
    
    ::nlpp::Rosenbrock func;

    for(int numVariables = 5; numVariables <= 10; numVariables += 1)
    {
        SCOPED_TRACE((std::string("Rosenbrock \t N: ") + std::to_string(numVariables)).c_str());

        convergenceTest(opt, func, ::nlpp::Vec::Constant(numVariables, 1.2));
    }
}

TEST_F(TrustRegionTest, DogLeg)
{
    SCOPED_TRACE("Dog Leg Test");

    ::nlpp::DogLeg<::nlpp::stop::GradientNorm<>> opt;
    opt.stop = ::nlpp::stop::GradientNorm<>(10000, 1e-4);
    
    ::nlpp::Rosenbrock func;

    for(int numVariables = 10; numVariables <= 100; numVariables += 10)
    {
        SCOPED_TRACE((std::string("Rosenbrock \t N: ") + std::to_string(numVariables)).c_str());

        convergenceTest(opt, func, ::nlpp::Vec::Constant(numVariables, 2.0));
    }
}

TEST_F(TrustRegionTest, IndefiniteDogLeg)
{
    SCOPED_TRACE("Indefinite Dog Leg Test");

    ::nlpp::IndefiniteDogLeg<::nlpp::stop::GradientNorm<>> opt;
    opt.stop = ::nlpp::stop::GradientNorm<>(10000, 1e-4);
    
    ::nlpp::Rosenbrock func;

    for(int numVariables = 10; numVariables <= 100; numVariables += 10)
    {
        SCOPED_TRACE((std::string("Rosenbrock \t N: ") + std::to_string(numVariables)).c_str());

        convergenceTest(opt, func, ::nlpp::Vec::Constant(numVariables, 2.0));
    }
}

TEST_F(TrustRegionTest, IterativeTR)
{
    SCOPED_TRACE("Iterative Trust Region Test");

    ::nlpp::IterativeTR<::nlpp::stop::GradientNorm<>> opt;
    opt.stop = ::nlpp::stop::GradientNorm<>(10000, 1e-4);
    
    ::nlpp::Rosenbrock func;

    for(int numVariables = 50; numVariables <= 100; numVariables += 10)
    {
        SCOPED_TRACE((std::string("Rosenbrock \t N: ") + std::to_string(numVariables)).c_str());

        convergenceTest(opt, func, ::nlpp::Vec::Constant(numVariables, 5.0));
    }
}


} // namespace