#include <type_traits>
#include <iostream>
#include <string>

using namespace std;


template <class T, class V>
concept Function = std::is_same_v<std::decay_t<V>, int> &&
(requires(const T& f, V v)
{
    { f(v) } -> std::floating_point;
} ||
requires(const T& f, V v)
{
    { f.function(v) } -> std::floating_point;
})
;


template <class V, Function<V> F>
void foo (const F& f, const V& v)
{
    // f(v);
    std::cout << "A\n";
}

template <class V, class F>
void foo (const F& f, const V& v)
{
    // f(v);
    std::cout << "B\n";
}

struct F1
{
    // float operator () (int) const { return 0.0; }
    float function (int) const { return 0.0; }
};

struct F2
{
    void operator () (int) const { }
};

#include <atomic>

int main ()
{
    foo(F1{}, 10);
    foo(F2{}, 10);

    std::cout << Function<F1, int> << "\n";
    std::cout << Function<F2, int> << "\n";

    std::atomic<int> x = 0;

    return 0;
}