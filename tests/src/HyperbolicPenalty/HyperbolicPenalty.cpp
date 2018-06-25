#include "gtest/gtest.h"

#include "HyperbolicPenalty/HyperbolicPenalty.h"

#include "TestFunctions/TreeBarTruss.h"


namespace
{

struct HyperbolicPenalty : public ::testing::Test
{
    virtual ~HyperbolicPenalty() {}

    virtual void SetUp () {}

    virtual void TearDown () {}
};



TEST_F(HyperbolicPenalty, PrecisionTest)
{
    nlpp::Vec xBest(2);
    xBest << 0.788675, 0.408248;

    double fxBest = 263.895843;


    nlpp::HyperbolicPenalty<> hp;

    nlpp::Vec x = hp(nlpp::TreeBarTruss::func, nlpp::TreeBarTruss::cons, nlpp::Vec::Constant(2, 0.5));
    

    ASSERT_LE((x - xBest).norm(), 0.1);
    ASSERT_LE(std::abs(nlpp::TreeBarTruss::func(x) - fxBest), 0.1);
}



} // namespace