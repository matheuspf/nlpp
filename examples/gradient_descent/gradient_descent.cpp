#include "gradient_descent/gradient_descent.hpp"
#include "TestFunctions/Rosenbrock.h"


int main ()
{
    nlpp::GradientDescent<> opt;

    nlpp::Rosenbrock func;
    nlpp::Vec x0 = nlpp::Vec::Constant(10, 2.0);

    nlpp::Vec res;
    
    handy::print(handy::benchmark([&]{
        res = opt(func, x0);
    }));

    handy::print(res.transpose());


    return 0;
}
