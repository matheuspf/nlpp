#include <type_traits>
#include <iostream>
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
    // nlpp::Rosenbrock func;
    // Eigen::VectorXd v = Eigen::VectorXd::Zero(2);
    
    // foo(func, v);
    // foo(goo, v);

    std::cout << VecType<Eigen::VectorX<float>> << "\n";
    std::cout << VecType<Eigen::VectorX<double>> << "\n";
    std::cout << VecType<Eigen::MatrixX<float>> << "\n";
    std::cout << VecType<Eigen::MatrixX<double>> << "\n";

    std::cout << "\n";

    std::cout << MatType<Eigen::VectorX<float>> << "\n";
    std::cout << MatType<Eigen::VectorX<double>> << "\n";
    std::cout << MatType<Eigen::MatrixX<float>> << "\n";
    std::cout << MatType<Eigen::MatrixX<double>> << "\n";

    std::cout << "\n";

    std::cout << VecType<decltype(Eigen::VectorX<float>{} + Eigen::VectorX<float>{})> << "\n";
    std::cout << VecType<decltype(Eigen::MatrixX<float>{} + Eigen::MatrixX<float>{})> << "\n";
    std::cout << MatType<decltype(Eigen::VectorX<float>{} + Eigen::VectorX<float>{})> << "\n";
    std::cout << MatType<decltype(Eigen::MatrixX<float>{} + Eigen::MatrixX<float>{})> << "\n";
    
    

    return 0;
}
