#pragma once

#include "../Helpers/Helpers.h"


namespace nlpp
{

template <int M = 5>
struct Shekel
{
    Shekel() : C(4, 10), b(10)
    {
        C << 4, 1, 8, 6, 3, 2, 5, 8, 6, 7,
             4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6,
             4, 1, 8, 6, 3, 2, 5, 8, 6, 7,
             4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6;
        b << 1, 2, 2, 4, 4, 6, 3, 7, 5, 5;
        b /= 10;
    }

    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x) const
    {
        impl::Scalar<V> res = 0.0;

        for(int i = 0; i < M; ++i)
        {
            impl::Scalar<V> r = 0.0;

            for(int j = 0; j < 4; ++j)
                r += std::pow(x(j) - C(j, i), 2);

            res += -1 / (r + b(i));
        }

        return res;
    }


    Mat C;
    Vec b;
};

} // namespace nlpp