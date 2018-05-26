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


    // auto func = [](const Vec& x){ return 10.0; };
    // auto grad1 = [](const Vec& x){ return x; };
    // auto grad2 = [](const Vec& x, Vec& g){};
    // auto funcGrad1 = [](const Vec& x){ return std::make_pair(10.0, Vec()); };
    // auto funcGrad2 = [](const Vec& x, Vec& g){ return 10.0; };

    // handy::print("IsFunction:\n");
    // handy::print("func: ", wrap::IsFunction<decltype(func), Vec>::value);
    // handy::print("grad1: ", wrap::IsFunction<decltype(grad1), Vec>::value);
    // handy::print("grad2: ", wrap::IsFunction<decltype(grad2), Vec>::value);
    // handy::print("funcGrad1: ", wrap::IsFunction<decltype(funcGrad1), Vec>::value);
    // handy::print("funcGrad2: ", wrap::IsFunction<decltype(funcGrad2), Vec>::value, "\n");
    
    // handy::print("IsGradient:\n");
    // handy::print("func: ", wrap::IsGradient<decltype(func), Vec>::value);
    // handy::print("grad1: ", wrap::IsGradient<decltype(grad1), Vec>::value);
    // handy::print("grad2: ", wrap::IsGradient<decltype(grad2), Vec>::value);
    // handy::print("funcGrad1: ", wrap::IsGradient<decltype(funcGrad1), Vec>::value);
    // handy::print("funcGrad2: ", wrap::IsGradient<decltype(funcGrad2), Vec>::value, "\n");

    // handy::print("IsFunctionGradient:\n");
    // handy::print("func: ", wrap::IsFunctionGradient<decltype(func), Vec>::value);
    // handy::print("grad1: ", wrap::IsFunctionGradient<decltype(grad1), Vec>::value);
    // handy::print("grad2: ", wrap::IsFunctionGradient<decltype(grad2), Vec>::value);
    // handy::print("funcGrad1: ", wrap::IsFunctionGradient<decltype(funcGrad1), Vec>::value);
    // handy::print("funcGrad2: ", wrap::IsFunctionGradient<decltype(funcGrad2), Vec>::value, "\n");

    return 0;
}