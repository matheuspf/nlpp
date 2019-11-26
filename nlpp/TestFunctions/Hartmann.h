#pragma once

#include "../helpers/helpers.hpp"


namespace nlpp
{

template <int N = 3>
struct Hartmann
{
    Hartmann () : a(4), A(4, N), P(4, N)
    {
        a << 1, 1.2, 3, 3.2;
        init();
    }

    template <class V>
    auto operator () (const Eigen::MatrixBase<V>& x) const
    {
        impl::Scalar<V> res = 0.0;

        for(int i = 0; i < 4; ++i)
        {
            impl::Scalar<V> ri = 0.0;

            for(int j = 0; j < N; ++j)
                ri += A(i, j) * std::pow(x(j) - P(i, j), 2);

            res += a(i) * std::exp(-ri);
        }
        
        return -res;
    }

    template <int N_ = N, std::enable_if_t<N_ == 3, int> = 0>
    void init ()
    {
        A << 3, 10, 30,
             0.1, 10, 35,
             3, 10, 30,
             0.1, 10, 35;

        P << 3689, 1170, 2673,
             4699, 4387, 7470,
             1091, 8732, 5547,
             381, 5743, 8828;

        P *= 1e-4;
    }

    template <int N_ = N, std::enable_if_t<N_ == 6, int> = 0>
    void init ()
    {
        A << 10, 3, 17, 3.5, 1.7, 8,
             0.05, 10, 17, 0.1, 8, 14,
             3, 3.5, 1.7, 10, 17, 8,
             17, 8, 0.05, 10, 0.1, 14;

        P << 1312, 1696, 5569, 124, 8283, 5886,
             2329, 4135, 8307, 3736, 1004, 9991,
             2348, 1451, 3522, 2883, 3047, 6650,
             4047, 8828, 8732, 5743, 1091, 381;

        P *= 1e-4;
    }


    Vec a;
    Mat A, P;
};

} // namespace nlpp