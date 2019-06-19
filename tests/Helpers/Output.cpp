#include <gtest/gtest.h>

#include "Helpers/Output.h"


namespace
{

// struct OutputTest : public ::testing::Test
// {
//     virtual ~OutputTest () {}

//     virtual void SetUp ()
//     {
//     }
// };


TEST(OutputTest, NoOutput)
{
    nlpp::GradientOptimizer<> opt;

    nlpp::out::Optimizer<0> out;
}


// TEST_F(FiniteDifferenceTest, GradientTest)
// {
//     SCOPED_TRACE("Finite Gradient Test");

//     handy::forEach(functions, gradients, names, variables, [&](auto& f, auto& g, auto& n, auto& v)
//     {
//         SCOPED_TRACE((std::string("Function: ") + n).c_str());
//         handy::RandDouble rng; nlpp::Vec x(v); for(int i = 0; i < 10; ++i)
//         {
//             std::for_each(x.data(), x.data() + x.size(), [&rng](auto& xi) { xi = rng(-10.0, 10.0); });
                
//             SCOPED_TRACE((std::string("X: ") + nlpp::impl::toString(x)).c_str());
            
//             auto finG = nlpp::fd::gradient(f);
            
//             testGradient(finG, g, x, 1e-4);
//         }
//     });
// }


} // namespace