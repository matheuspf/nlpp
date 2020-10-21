#include <gtest/gtest.h>

#include "utils/wrappers/domain.hpp"

namespace
{

struct Domain : public ::testing::Test
{
    virtual ~Domain () {}

    virtual void SetUp () {}
    virtual void TearDown () {}


    template <class D, class V>
    static void testDomain (const D& domain, const Eigen::MatrixBase<V>& x)
    {
        if constexpr(D::HasStart)
            EXPECT_FALSE(nlpp::isNan(domain.x0) && nlpp::isInf(domain.x0));
        
        if constexpr(D::HasBounds)
        {
            EXPECT_TRUE(domain.within(x));

            for(int i = 0; i < x.rows(); ++i)
                EXPECT_TRUE(domain.within(x, i));
        }

        if constexpr(D::HasLinearInequalities)
            EXPECT_LE(domain.ineqs(x).sum(), 0.0);

        if constexpr(D::HasLinearEqualities)
            EXPECT_LE(domain.eqs(x).sum(), 0.0);
    }
};


TEST_F(Domain, SimpleTest)
{
    SCOPED_TRACE("Domain Simple Test");

    using V = nlpp::Vec;

    V x0 = V::Constant(2, 1.0);
    V lb = V::Constant(2, 0.0);
    V ub = V::Constant(2, 10.0);
    nlpp::impl::Plain2D<V> A = nlpp::impl::Plain2D<V>::Constant(2, 2, 1.0);
    V b = V::Constant(2, 10.0);
    nlpp::impl::Plain2D<V> Aeq = nlpp::impl::Plain2D<V>::Constant(2, 2, 1.0);
    V beq = V::Constant(2, 10.0);

    nlpp::wrap::Domain<V, nlpp::wrap::Conditions::Start> d1(x0);
    nlpp::wrap::Domain<V, nlpp::wrap::Conditions::Start | nlpp::wrap::Conditions::Bounds> d2(x0, lb, ub);
    nlpp::wrap::Domain<V, nlpp::wrap::Conditions::LinearInequalities> d3(nlpp::impl::Plain2D<V>(2, 2), b);
    nlpp::wrap::Domain<V, nlpp::wrap::Conditions::FullDomain> d4(x0);

    auto vecs = {
        V::Constant(2, 2.0),
        V::Constant(2, 3.0),
        V::Constant(2, 0.0),
    };

    for(const auto& x : vecs)
    {
        SCOPED_TRACE(std::string("Vector ") + nlpp::impl::toString(x.transpose()));

        testDomain(d1, x);
        testDomain(d2, x);
        testDomain(d3, x);
        testDomain(d4, x);
    }
}


} // namespace