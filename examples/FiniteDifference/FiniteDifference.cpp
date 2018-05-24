///[FiniteDifference snippet]

#include <iostream>
#include "Helpers/FiniteDifference.h"

using namespace Eigen;
using namespace cppnlp;


int main ()
{
    auto func = [](const VectorXd& x)
    {
        return pow(x[0], 3) + 2 * x[1] * x[1] + 5 * x[0] * x[1];
    };

    auto grad = fd::gradient(func);

    auto hess = fd::hessian(func);

    VectorXd x(2);
    x << 3.0, 2.0;


    handy::print(grad(x).transpose());

    handy::print(hess(x));


    return 0;   
}

///[FiniteDifference snippet]