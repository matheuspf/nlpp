#include "utils/wrappers/functions.hpp"
#include "utils/finite_difference_dec.hpp"

using namespace nlpp;


using Vec = Eigen::VectorXd;

struct Func
{
    template <class V>
    impl::Scalar<V> function (const Eigen::MatrixBase<V>& x) const
    {
        return x(0) + x(1);
    }
};

struct Grad1
{
    template <class V>
    auto gradient (const Eigen::MatrixBase<V>& x) const
    {
        return 2 * x;
    }
};

struct Grad2
{
    template <class V>
    void operator () (const Eigen::MatrixBase<V>& x, typename V::PlainObject& g) const
    {
        g = 2 * x;
    }
};

struct FuncGrad1
{
    template <class V>
    auto operator () (const Eigen::MatrixBase<V>& x) const
    {
        return std::make_pair(Func{}(x), Grad1{}(x));
    }
};

struct FuncGrad2
{
    template <class V>
    impl::Scalar<V> operator () (const Eigen::MatrixBase<V>& x, typename V::PlainObject& g, bool calcGrad) const
    {
        if(calcGrad)
            Grad2{}(x, g);

        return Func{}.function(x);
    }
};

struct Hessian
{
    // template <class V, class U>
    // impl::Plain<V> operator () (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e)
    // {
    //     return x + e;
    // }

    template <class V>
    impl::Plain2D<V> operator () (const Eigen::MatrixBase<V>& x) const
    {
        return x * x.transpose();
    }
};


auto genericFunction = [](const auto& x)
{
    return x(0) + x(1);
};


int main ()
{
    Eigen::Vector2d x(2, 3), gx(2, 3);
    Eigen::VectorXf y(2), gy(2); y << 2, 3;

    auto func = wrap::function(Func{});

    auto grad1 = wrap::gradient(Grad1{});
    auto grad2 = wrap::gradient(Grad2{});

    auto funcGrad1 = wrap::funcGrad(Func{}, Grad2{});
    auto funcGrad2 = wrap::funcGrad(FuncGrad2{});
    // auto funcGrad3 = wrap::funcGrad(Func{});
    auto funcGrad3 = wrap::fd::funcGrad(Func{});

    auto hess = wrap::hessian(Hessian{});

    auto funcs = wrap::fd::functions<wrap::Conditions::AllFunctions, Eigen::VectorXf>(genericFunction);

    handy::print(func(x), func(y));
    handy::print(func(x+x), func(y+y), "\n");

    handy::print(grad1(x).transpose(), grad1(y).transpose());
    handy::print(grad1(x+x).transpose(), grad1(y+y).transpose());
    grad2(x, gx), grad2(y, gy); handy::print(gx.transpose(), gy.transpose());
    grad2(x+x, gx), grad2(y+y, gy); handy::print(gx.transpose(), gy.transpose(), "\n");

    handy::print(std::get<0>(funcGrad1(x+x)), std::get<0>(funcGrad1(y+y)));
    handy::print(std::get<1>(funcGrad1(x)).transpose(), std::get<1>(funcGrad1(y)).transpose());
    handy::print(funcGrad2(x, gx), funcGrad2(y, gy));
    funcGrad2(x+x, gx), funcGrad2(y+y, gy); handy::print(gx.transpose(), gy.transpose());
    handy::print(funcGrad1.function(y+y), funcGrad1.gradient(x+x).transpose());
    handy::print(funcGrad2.function(y+y), funcGrad2.gradient(x+x).transpose(), "\n");

    // //handy::print(std::get<0>(funcGrad3(x+x)), std::get<1>(funcGrad3(y+y)).transpose());
    handy::print(funcGrad3(x+x, gx), gx.transpose(), "\n");

    handy::print(hess.hessianDir(x+x, x).transpose(), "\n");
    std:: cout << hess(x+x).transpose() << "\n\n";

    handy::print(funcs.function(x+x), funcs.function(y+y));
    handy::print(funcs.gradient(x+x).transpose(), funcs.gradient(y+y).transpose());
    handy::print(funcs.hessianDir(y+y, x).transpose());
    handy::print(std::get<0>(funcs(x+x)), std::get<0>(funcs(y+y)));

    return 0;
}