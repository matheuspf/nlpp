#include "LineSearch/Dynamic/Dynamic.h"

#include "GradientDescent/GradientDescent.h"

#include "TestFunctions/Rosenbrock.h"


using namespace nlpp;




int main ()
{
    using Func = Rosenbrock;
    using LS = DynamicLineSearch<Func>;
    
    Rosenbrock func;

    Vec x0 = Vec::Constant(50, 1.2);

	GradientDescent<LS> gd1, gd2;

    gd1.lineSearch = LS("Goldstein");
    gd2.lineSearch = LS("StrongWolfe");

    auto x1 = gd1(func, x0);
    auto x2 = gd2(func, x0);

    handy::print(x1.transpose(), "\n", x2.transpose());



    return 0;
}