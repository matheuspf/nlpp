#include <iostream>
#include "Helpers/FiniteDifference.h"

using namespace cppnlp;






int main ()
{
    auto func = [](const Vec& x)
    {
        return pow(x[0], 3) + 2 * x[1] * x[1];
    };

    auto diff = gradientFD(func);


    Vec x(2); x << 3.0, 2.0;

    handy::print(diff(x + x));


    return 0;   
}