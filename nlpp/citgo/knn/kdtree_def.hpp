#pragma once

#include "../../helpers/helpers.hpp"

namespace nlpp
{

struct KDTreeKNN
{
    template <class V>
    Matu operator() (const Eigen::MatrixBase<V>& X, int k) const;

    int max_leaves = 10;
    nanoflann::SearchParams searchParams{10};
};


} // namespace nlpp
