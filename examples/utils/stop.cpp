#include "utils/stop.hpp"

using namespace nlpp;

struct Dummy {};


int main ()
{
    Vec x = Vec::Constant(3, 2.0);
    Vec gx = Vec::Constant(3, 4.0);
    Vec::Scalar fx = 1.0;

    using Float = types::Float;

    int maxIterations = 1e4;
    Float xTol = 1e-4;
    Float fTol = 1e-4;
    Float gTol = 1e-4;

    stop::Optimizer<false, double> opt0(maxIterations, xTol, fTol);
    stop::Optimizer<true, double> opt1(maxIterations, xTol, fTol);

    std::cout << opt0(Dummy{}, x, fx) << "\t" << opt0(Dummy{}, x, fx) << "\t" << opt0(Dummy{}, x, fx + 2*xTol) << "\n";
    std::cout << opt1(Dummy{}, x, fx) << "\t" << opt1(Dummy{}, x, fx) << "\t" << opt1(Dummy{}, x, fx + 2*xTol) << "\n";


    stop::GradientOptimizer<false, double> gopt0(maxIterations, xTol, fTol, gTol);
    stop::GradientOptimizer<true, double> gopt1(maxIterations, xTol, fTol, gTol);

    std::cout << gopt0(Dummy{}, x, fx, gx) << "\t" << gopt0(Dummy{}, x, fx, gx) << "\t" << gopt0(Dummy{}, x, fx + 2*xTol, gx) << "\n";
    std::cout << gopt1(Dummy{}, x, fx, gx) << "\t" << gopt1(Dummy{}, x, fx, gx) << "\t" << gopt1(Dummy{}, x, fx + 2*xTol, gx) << "\n";


    stop::GradientNorm gnopt(maxIterations, gTol);

    std::cout << gnopt(Dummy{}, x, fx, gx) << "\t" << gnopt(Dummy{}, x, fx, Vec::Constant(3, gTol)) << "\n";


    return 0;
}