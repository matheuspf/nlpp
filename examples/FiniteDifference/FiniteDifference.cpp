#include <iostream>
#include "Helpers/FiniteDifference.h"

using namespace cppnlp;






int main ()
{
    // auto func = [](const Vec& x)
    // {
    //     return pow(x[0], 3) + 2 * x[1] * x[1];
    // };

    auto func = [](double x)
    {
        return 2 * pow(x, 3);
    };


    //auto diff = gradientFD(func);
    auto diff = hessianFD(func);
    //auto diff = hessianFD(gradientFD(func), SimpleStep<double>{1e-2});


    //Vec x(2); x << 3.0, 2.0;
    //Eigen::Vector2d x(3.0, 2.0);
    double x = 3.0;

    handy::print(diff(x));


    return 0;   
}