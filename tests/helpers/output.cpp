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


// 


} // namespace