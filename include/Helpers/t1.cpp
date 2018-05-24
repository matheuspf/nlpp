#include "Gradient.h"

using namespace cppnlp;


using Vec = Eigen::VectorXd;

struct Func
{
    double operator () (const Vec& x)
    {
        return x[0] + x[1];
    }
};

struct Grad
{
    Vec operator () (const Vec& x)
    {
        return 2 * x;
    }
};

struct Grad2
{
    void operator () (const Vec& x, Vec& g)
    {
        g = 2 * x;
    }
};

struct FuncGrad
{
    auto operator () (const Vec& x)
    {
        return std::make_pair(Func{}(x), Grad{}(x));
    }
};

struct FuncGrad2
{
    double operator () (const Vec& x, Vec& g)
    {
        Grad2{}(x, g);

        return Func{}(x);
    }
};



int main ()
{
    auto func1 = wrap::functionGradient(Func{}, [](const Vec& x) -> Vec { return 2 * x; });

    Vec x(2); x << 1.0, 2.0;
    Vec g12(2), g22(2);

    auto [f11, g11] = func1(x);
    
    double f12 = func1(x, g12);

    handy::print(f11, "   ", g11.transpose());
    handy::print(f12, "   ", g12.transpose());


    auto func2 = wrap::functionGradient(FuncGrad{});

    auto [f21, g21] = func2(x);

    double f22 = func2(x, g22);

    handy::print(f21, "   ", g21.transpose());
    handy::print(f22, "   ", g22.transpose());

    return 0;
}