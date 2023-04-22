#include <iostream>
#include <type_traits>

using namespace std;


template <class T>
concept Int = std::same_as<T, int>;

template <class T>
concept Float = std::same_as<T, float>;


template <Int... T, Float... U>
// requires (std::is_same_v<U, float> && ...)
void foo (T&&... t, U&&... u)
{
    std::cout << sizeof...(t) << "\n";
    std::cout << sizeof...(u) << "\n";
}


int main()
{
    foo(10, 20, 10.0, 20.0);
    return 0;
}