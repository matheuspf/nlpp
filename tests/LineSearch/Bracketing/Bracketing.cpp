#include "gtest/gtest.h"

#include "LineSearch/Bracketing/Bracketing.h"


namespace
{

struct Bracketing : ::testing::Test
{
    virtual ~Bracketing ()
    {
    }

    virtual void SetUp ()
    {
    }

    virtual void TearDown ()
    {
    }


	static double func (double x)
	{
		return (x - 3) * pow(x, 3) * pow(x - 6, 4);
	}

    
    void testRange (double l, double r)
    {
        SCOPED_TRACE("testRange");

        std::tie(x, y, z) = bc(func, l, r);

        ASSERT_LE(x, y);
        ASSERT_LE(y, z);

        ASSERT_LE(func(y), func(x));
        ASSERT_LE(func(y), func(z));
    }


    nlpp::Bracketing<> bc;

    double x, y, z;
};


TEST_F(Bracketing, BracketingTest)
{
    testRange(0.0, 3.0);
    testRange(-100.0, 100.0);
    testRange(-1.0, 1.0);
}


} // namespace