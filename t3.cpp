#include <bits/stdc++.h>
#include "nlpp/TestFunctions/Rosenbrock.h"

using namespace std;


template <class V>
using Plain = typename std::decay_t<V>::PlainObject;

template <class V>
concept MatrixBaseType = std::derived_from<Plain<V>, Eigen::MatrixBase<Plain<V>>>;

template <class V>
concept VecType = MatrixBaseType<V> && Plain<V>::ColsAtCompileTime == 1;

template <class V>
concept MatType = MatrixBaseType<V> && Plain<V>::ColsAtCompileTime != 1;


template <class F, class V>
concept FuncBase = VecType<V> && requires(const F& f, const V& v)
{
    { f(v) } -> std::floating_point;
};

template <class F, typename... Args>
concept FuncHelper = (FuncBase<F, Eigen::VectorX<Args>> || ...);

template <class F>
concept Func = FuncHelper<F, float, double, long double>;



template <Func F, class V>
auto foo (const F& f, const V& v)
{
    return f(v);
}




float goo (const Eigen::Vector2<double>& x)
{
    return 10;
}


template <class> class Prt;


int main ()
{
    nlpp::Rosenbrock func;
    Eigen::VectorXd v = Eigen::VectorXd::Zero(2);
    
    foo(func, v);
    foo(goo, v);
    

    return 0;
}
