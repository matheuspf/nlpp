#include <gtest/gtest.h>

#include "utils/wrappers/constraints.hpp"

namespace
{

struct ConstraintsTest : public ::testing::Test
{
    virtual ~ConstraintsTest () {}

    virtual void SetUp ()
    {
        vecs = {
            nlpp::Vec::Constant(2, 20.0),
            nlpp::Vec::Constant(3, 50.0),
        };
    }

    virtual void TearDown ()
    {
        vecs.clear();
    }

    template <nlpp::wrap::Conditions Cond, class V, class... Fs>
    static void testFunction (const Eigen::MatrixBase<V>& x, const Fs&... fs)
    {
        using C = nlpp::wrap::Constraints<Cond, Fs...>;

        auto c = C(fs...);

        if constexpr(C::HasIneqs & C::HasEqs)
            EXPECT_EQ(c(x), std::make_pair(c.ineqs(x), c.eqs(x)));

        if constexpr(C::HasIneqs)
            EXPECT_GT(c.ineqs(x).sum(), 0.0);
        
        if constexpr(C::HasEqs)
            EXPECT_GT(c.eqs(x).sum(), 0.0);
    }

    std::vector<nlpp::Vec> vecs;


    struct F1
    {
        template <class V>
        nlpp::impl::Plain<V> ineqs (const Eigen::MatrixBase<V>& x) const
        {
            return x - nlpp::impl::Plain<V>::Constant(x.rows(), 1.0);
        }
    };

    struct F2
    {
        template <class V>
        nlpp::impl::Plain<V> eqs (const Eigen::MatrixBase<V>& x) const
        {
            return x - nlpp::impl::Plain<V>::Constant(x.rows(), 2.0);
        }
    };

    struct F3
    {
        template <class V>
        nlpp::impl::Plain<V> operator() (const Eigen::MatrixBase<V>& x) const
        {
            return x - nlpp::impl::Plain<V>::Constant(x.rows(), 3.0);
        }
    };
};


TEST_F(ConstraintsTest, BaseTest)
{
    using nlpp::wrap::Conditions;

    SCOPED_TRACE("Constraints Base Test");

    auto f4 = [](const auto& x)
    {
        return x - nlpp::impl::Plain<std::decay_t<decltype(x)>>::Constant(x.rows(), 4.0);
    };

    auto f5 = [](const auto& x)
    {
        return std::make_pair(F1{}.ineqs(x), F2{}.eqs(x));
    };


    for(const auto& x : vecs)
    {
        SCOPED_TRACE(std::string("Vector ") + nlpp::impl::toString(x));

        testFunction<Conditions::NLInequalities>(x, F1{});
        testFunction<Conditions::NLEqualities>(x, F2{});
        testFunction<Conditions::NLInequalities | Conditions::NLEqualities>(x, F1{}, F2{});
        testFunction<Conditions::NLEqualities>(x, F3{});
        testFunction<Conditions::NLInequalities | Conditions::NLEqualities>(x, F3{}, F3{});
        testFunction<Conditions::NLInequalities>(x, f4);
        testFunction<Conditions::NLInequalities | Conditions::NLEqualities>(x, f4, f4);
        testFunction<Conditions::NLInequalities | Conditions::NLEqualities>(x, f5);
        testFunction<Conditions::Empty>(x, F1{});
        testFunction<Conditions::NLInequalities>(x, F1{}, F2{}, F3{});
    }
}

} // namespace