#include <type_traits>
#include <iostream>
#include <string>

using namespace std;

#define RESULT_OF_T(NAME) \
template <class F, class... Args> \
using Invoke_NAME = decltype(std::declval<F>.func(std::declval<Args>()...)); \
\
template <class VOID, class F, class... Args> \
struct IsValid_NAME { enum { valid = false }; }; \
\
template <class F, class... Args> \
struct IsValid_NAME<Invoke_NAME<F, Args...>, F, Args...> { enum { valid = true }; }; \
\
template <class F, class... Args> \
struct Result_of_NAME : std::enable_if<IsValid_NAME<F, Args...>::value, decltype(std::declval<F>.func(std::declval<Args>()...))> {}; \
\
template <class F, class... Args> \
using Result_of_NAME_t = typename Result_of_NAME<F, Args...>::type; \


RESULT_OF_T(func)


struct X
{
    void func (int) {}
};

struct Y
{
    void func (std::string) {}
};



template <class T, class U = Result_of_NAME_t<T, int>>
void test () { std::cout << "A\n"; }

template <class T, class U = Result_of_NAME_t<T, std::string>>
void test () { std::cout << "B\n"; }


int main ()
{
    test<X>();
    // test<Y>();

    return 0;
}