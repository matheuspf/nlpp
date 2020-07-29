#pragma once

#include "../../helpers/helpers.hpp"

namespace nlpp
{

template <class D1, class D2>
Eigen::Matrix<impl::Scalar<D1>, D1::ColsAtCompileTime, D2::ColsAtCompileTime> pdist2(const Eigen::MatrixBase<D1>& X, const Eigen::MatrixBase<D2>& Y);

struct DefaultKNN
{
    DefaultKNN (bool ordered = true) : ordered(ordered) {}

    template <class V>
    Matu operator() (const Eigen::MatrixBase<V>& X, int k) const;

    bool ordered;
};

}; // namespace nlpp