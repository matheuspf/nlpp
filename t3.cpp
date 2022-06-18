#include <type_traits>
#include <iostream>
#include <string>
#include <Eigen/Dense>

using namespace std;


template <class T>
// concept Vector = std::is_convertible_v<std::decay_t<T>, Eigen::VectorXd>;
concept Vector = std::is_convertible_v<decltype(&std::declval<std::decay_t<T>>()), decltype(&std::declval<Eigen::MatrixBase<std::decay_t<T>>>()) >;


// template <class T, class V>
// concept Function = std::is_same_v<std::decay_t<V>, int> &&
// (requires(const T& f, V v)
// {
//     { f(v) } -> std::floating_point;
// } ||
// requires(const T& f, V v)
// {
//     { f.function(v) } -> std::floating_point;
// })
// ;

int main ()
{
    std::cout << Vector<int> << "\n";
    std::cout << Vector<Eigen::VectorXd> << "\n";

    return 0;
}