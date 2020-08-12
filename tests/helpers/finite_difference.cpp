#include "gtest/gtest.h"

#include "utils/finite_difference_dec.hpp"


namespace
{

struct FiniteDifferenceTest : public ::testing::Test
{
    using Function = std::function<double(const nlpp::Vec&)>;
    using Gradient = std::function<nlpp::Vec(const nlpp::Vec&)>;
    using Hessian = std::function<nlpp::Mat(const nlpp::Vec&)>;

    virtual ~FiniteDifferenceTest () {}

    virtual void SetUp ()
    {
        functions.emplace_back([](const nlpp::Vec& x)
        {
            return std::pow(x[0], 4) + 5 * std::pow(x[1], 2);
        });

        gradients.emplace_back([](const nlpp::Vec& x) -> nlpp::Vec
        {
            nlpp::Vec g(x.size());

            g[0] = 4 * std::pow(x[0], 3);
            g[1] = 10 * x[1];

            return g;
        }); 
        
        hessians.emplace_back([](const nlpp::Vec& x) -> nlpp::Mat
        {
            nlpp::Mat h(x.size(), x.size());

            h(0, 0) = 12 * std::pow(x[0], 2);
            h(1, 1) = 10;

            return h;
        }); 

        names.emplace_back("x[0]^4 + 5*x[1]^2");
        variables.emplace_back(2);
    }


    template <class FinFunc, class ExactFunc>
    void testGradient (FinFunc finFunc, ExactFunc exactFunc, const nlpp::Vec& x, double tol = 1e-4)
    {
        auto finVal = finFunc(x);
        auto exactVal = exactFunc(x);

        double diff = (finVal - exactVal).squaredNorm();

        auto testCallBack = [&]
        {
            if(diff < tol)
                return ::testing::AssertionSuccess();
            
            else
                return ::testing::AssertionFailure() << "Finite estimation failed with error: " << diff << 
                                                        ". Expected to be less than: " << tol;
        };

        EXPECT_TRUE(testCallBack());
    }


    std::vector<Function> functions;
    std::vector<Gradient> gradients;
    std::vector<Hessian> hessians;
    std::vector<std::string> names;
    std::vector<int> variables;
};


TEST_F(FiniteDifferenceTest, GradientTest)
{
    SCOPED_TRACE("Finite Gradient Test");

    handy::forEach(functions, gradients, names, variables, [&](auto& f, auto& g, auto& n, auto& v)
    {
        SCOPED_TRACE((std::string("Function: ") + n).c_str());
        handy::RandDouble rng; nlpp::Vec x(v); for(int i = 0; i < 10; ++i)
        {
            std::for_each(x.data(), x.data() + x.size(), [&rng](auto& xi) { xi = rng(-10.0, 10.0); });

            SCOPED_TRACE((std::string("X: ") + nlpp::impl::toString(x)).c_str());

            auto finG = nlpp::fd::gradient(f);
            
            testGradient(finG, g, x, 1e-4);
        }
    });
}


// TEST_F(FiniteDifferenceTest, HessianTest)
// {
//     SCOPED_TRACE("Finite Hessian Test\n");

//     handy::forEach(functions, hessians, names, variables, [&](auto& f, auto& h, auto& n, auto& v)
//     {
//         SCOPED_TRACE((std::string("Function: ") + n).c_str());
        
//         handy::RandDouble rng;

//         nlpp::Vec x(v);

//         for(int i = 0; i < 10; ++i)
//         {
//             std::for_each(x.data(), x.data() + x.size(), [&rng](auto& xi){ xi = rng(-10.0, 10.0); });

//             SCOPED_TRACE((std::string("X: ") + nlpp::impl::toString(x)).c_str());

//             auto finH = nlpp::fd::hessian(f);

//             testGradient(finH, h, x, 1e-3);
//         }
//     });
// }


} // namespace