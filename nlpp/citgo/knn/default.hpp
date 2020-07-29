#pragma once

#include "default_def.hpp"


namespace nlpp
{

template <class D1, class D2>
Eigen::Matrix<impl::Scalar<D1>, D1::ColsAtCompileTime, D2::ColsAtCompileTime> pdist2(const Eigen::MatrixBase<D1>& X, const Eigen::MatrixBase<D2>& Y)
{
    // (x - y)^(x - y) == x^x + y^y - 2x^y
    return X.colwise().squaredNorm().transpose() * D1::Ones(1, Y.cols()) +
           D2::Ones(X.cols(), 1) * Y.colwise().squaredNorm() -
           2 * X.transpose() * Y;
}


template <class V>
Matu DefaultKNN::operator() (const Eigen::MatrixBase<V>& X, int k) const
{
    auto dists = pdist2(X, X);
    Matu knn(k+1, X.cols());
    std::vector<std::size_t> idx(X.cols());

    for(int i = 0; i < X.cols(); ++i)
    {
        std::iota(idx.begin(), idx.end(), 0);

        if(ordered)
            std::sort(idx.begin(), idx.end(), [i, &dists](int x, int y){ return dists(x, i) < dists(y, i); });

        else
            std::nth_element(idx.begin(), idx.begin() + k + 1, idx.end(), [i, &dists](int x, int y){ return dists(x, i) < dists(y, i); });

        std::copy(idx.begin(), idx.begin() + k + 1, &knn(0, i));
    }

    return knn;
}


}; // namespace nlpp