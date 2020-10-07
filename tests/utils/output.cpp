#include <gtest/gtest.h>

#include "utils/output.hpp"


namespace
{

TEST(OutputTest, SimpleTest)
{
    using namespace nlpp;

    using V = Vec;
    using Scalar = impl::Scalar<V>;

    V x = V::Constant(3, 2.0);
    V gx = V::Constant(3, 4.0);
    Scalar fx = 1.0;

    std::vector<V> vx;
    std::vector<Scalar> vfx;

    out::Optimizer<2, double> opt(vx, vfx);
    opt(nullptr, x, fx);

    ASSERT_EQ(vx.size(), 1);
    ASSERT_EQ(vfx.size(), 1);

    EXPECT_EQ(vx[0], x);
    EXPECT_EQ(vfx[0], fx);

    vx.clear();
    vfx.clear();
    std::vector<V> vgx;

    out::GradientOptimizer<2, double> gopt(vx, vfx, vgx);
    gopt(nullptr, x, fx, gx);

    ASSERT_EQ(vx.size(), 1);
    ASSERT_EQ(vfx.size(), 1);
    ASSERT_EQ(vgx.size(), 1);

    EXPECT_EQ(vx[0], x);
    EXPECT_EQ(vfx[0], fx);
    EXPECT_EQ(vgx[0], gx);
}

// 


} // namespace