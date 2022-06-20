#include <bits/stdc++.h>

using namespace std;



template <class> class Prt;


// template <class T>
// void foo (T t)
// {
//     Prt<T>{};
// }


template <class...>
struct Empty
{
    template <class T=nullptr_t>
    Empty(const std::initializer_list<T>&) {}
};


template <class T=nullptr_t>
void foo (const std::initializer_list<T>&)
{
}


void goo(Empty<>, int, std::string, Empty<>) {}


int main ()
{
    foo({});
    goo({}, 10, " ", {});

    return 0;
}
