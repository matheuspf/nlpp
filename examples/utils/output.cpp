#include "utils/output.hpp"

using namespace nlpp;

struct Dummy {};


int main ()
{
    Vec x = Vec::Constant(3, 2.0);
    Vec gx = Vec::Constant(3, 4.0);
    Vec::Scalar fx = 1.0;

    out::Optimizer<0, double> opt0;
    out::Optimizer<1, double> opt1;
    out::Optimizer<2, double> opt2;

    opt0(Dummy{}, x, fx);

    opt1(Dummy{}, x, fx);

    opt2(Dummy{}, x, fx);
    std::cout << opt2.vX[0].transpose() << "\t" << opt2.vFx[0] << "\n";


    out::GradientOptimizer<0, double> gopt0;
    out::GradientOptimizer<1, double> gopt1;
    out::GradientOptimizer<2, double> gopt2;

    gopt0(Dummy{}, x, fx, gx);

    gopt1(Dummy{}, x, fx, gx);

    gopt2(Dummy{}, x, fx, gx);
    std::cout << gopt2.vX[0].transpose() << "\t" << gopt2.vFx[0] << "\t" << gopt2.vGx[0].transpose() << "\n";


    return 0;
}
