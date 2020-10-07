#include "utils/output.hpp"

using namespace nlpp;

struct Dummy {};


int main ()
{
    using V = Vec;
    using Scalar = impl::Scalar<V>;

    V x = V::Constant(3, 2.0);
    V gx = V::Constant(3, 4.0);
    Scalar fx = 1.0;


    out::Optimizer<0, double> opt0;
    out::Optimizer<1, double> opt1;

    std::vector<V> vx;
    std::vector<Scalar> vfx;
    out::Optimizer<2, double> opt2(vx, vfx);

    opt0(Dummy{}, x, fx);

    opt1(Dummy{}, x, fx);

    opt2(Dummy{}, x, fx);
    std::cout << vx[0].transpose() << "\t" << vfx[0] << "\n";


    out::GradientOptimizer<0, double> gopt0;
    out::GradientOptimizer<1, double> gopt1;

    vx.clear();
    vfx.clear();
    std::vector<V> vgx;
    out::GradientOptimizer<2, double> gopt2(vx, vfx, vgx);

    gopt0(Dummy{}, x, fx, gx);

    gopt1(Dummy{}, x, fx, gx);

    gopt2(Dummy{}, x, fx, gx);
    std::cout << vx[0].transpose() << "\t" << vfx[0] << "\t" << vgx[0].transpose() << "\n";


    return 0;
}
