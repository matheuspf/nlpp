#include "line_search/goldstein/goldstein.hpp"

using namespace nlpp;


auto bowl (const Vec& x)
{
    return pow(x[0] - 3, 2) + pow(x[1] - 3, 2);
}


int main ()
{
    ls::Goldstein<> gold;

    auto ff = wrap::fd::funcGrad(&bowl);

    Vec x(2); x << 1, 1;
    Vec d(2); d << 1, 1;

    auto r = gold(ff, x, d);

    std::cout << r << "\n";
    std::cout << ff.function(x + r * d) << "\n";


    return 0;
}