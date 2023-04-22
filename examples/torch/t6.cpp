#include <iostream>
#include <algorithm>
#include "nlpp/utils/wrappers/helpers.hpp"


using namespace std;

template <class F, class V>
concept FuncTypeBase = requires (const F& f, const V& v)
{
    { f(v) } -> std::floating_point;
};

template <typename T>
concept CC = std::floating_point<T>;


template <template <class> class T>
void ff () {}


float foo (int) { return 0; }
int goo (int) { return 0; }



template <class... Args>
constexpr int test ()
{
    constexpr std::array<bool, sizeof...(Args)> arr = { FuncTypeBase<Args, int>... };
    constexpr std::size_t idx = std::ranges::max_element(arr) - std::ranges::begin(arr);

    if constexpr (idx == 0 && arr[0] == 0)
        return 5;

    return idx;
}


int main()
{
    // std::cout << test<decltype(goo), decltype(goo), decltype(goo)>() << "\n";

    std::array<int, test<decltype(goo), decltype(goo), decltype(goo)>()> vv;

    std::cout << vv.size() << "\n";

    return 0;
}