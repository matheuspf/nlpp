#include "SR1.h"

#include "../../TestFunctions/Rosenbrock.h"


int main ()
{
    SR1<> sr1;

    Vec x = Vec::Constant(5, 5.0);
    //Vec x(2); x << -1.2, 1;

    x = sr1(Rosenbrock(), x);

    DB(x.transpose());

    return 0;
}