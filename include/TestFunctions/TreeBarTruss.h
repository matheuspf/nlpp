#ifndef OPT_TREE_BAR_TRUSS
#define OPT_TREE_BAR_TRUSS

#include "../Modelo.h"


struct TreeBarTruss
{
    static double func (const Vec& x)
    {
        return (2.0 * sqrt(2.0) * x(0) + x(1)) * 100.0;
    }

    static const Vec& cons (const Vec& x)
    {
        static Vec r(7);

        r(0) = -x(0);
        r(1) = -x(1);
        r(2) = x(0) - 1.0;
        r(3) = x(1) - 1.0;

        double l = 100.0;
        double P = 2.0;
        double sig = 2.0;

        r(4) = ((sqrt(2) * x(0) + x(1)) / (sqrt(2) * pow(x(0), 2) + 2 * x(0) * x(1))) * P - sig;

        r(5) = (x(1) / (sqrt(2) * pow(x(0), 2) + 2 * x(0) * x(1))) * P - sig;

        r(6) = (1.0 / (x(0) + sqrt(2) * x(1))) * P - sig;
        

        return r;
    }
};



#endif // OPT_TREE_BAR_TRUSS