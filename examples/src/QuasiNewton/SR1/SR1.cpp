#include "QuasiNewton/SR1/SR1.h"

#include "TestFunctions/Rosenbrock.h"


using namespace nlpp;


int main ()
{
    SR1<> sr1;

    Vec x = Vec::Constant(5, 5.0);
    //Vec x(2); x << -1.2, 1;

    x = sr1(Rosenbrock(), x);

    handy::print(x.transpose());

    return 0;
}