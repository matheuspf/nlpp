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

    std::cout  << opt0(nullptr, x, fx) << "\t" << opt0(nullptr, x, fx) << "\t\t\t" << opt0(nullptr, x, fx + 2*fTol) << "\n";
    std::cout  << opt1(nullptr, x, fx) << "\t" << opt1(nullptr, x, fx) << "\t" << opt1(nullptr, x, fx + 2*fTol) << "\n\n";

    stop::GradientOptimizer<false, double> gopt0(maxIterations, xTol, fTol, gTol);
    stop::GradientOptimizer<true, double> gopt1(maxIterations, xTol, fTol, gTol);

    std::cout << gopt0(nullptr, x, fx, gx) << "\t" << gopt0(nullptr, x, fx, gx) << "\t" << gopt0(nullptr, x, fx + 2*fTol, gx) << "\n";
    std::cout << gopt1(nullptr, x, fx, gx) << "\t" << gopt1(nullptr, x, fx, gx) << "\t" << gopt1(nullptr, x, fx + 2*fTol, gx) << "\n\n";


    stop::GradientNorm gnopt(maxIterations, gTol);

    std::cout << gnopt(nullptr, x, fx, gx) << "\t" << gnopt(nullptr, x, fx, Vec::Constant(3, gTol)) << "\n";

    return 0;
}