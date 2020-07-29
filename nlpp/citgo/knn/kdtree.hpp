#pragma once

#include "kdtree_def.hpp"

namespace nlpp
{

template <class V>
Matu KDTreeKNN::operator() (const Eigen::MatrixBase<V>& X, int k) const
{
    const auto& Xt = X.transpose();
    nanoflann::KDTreeEigenMatrixAdaptor kdtree(Xt.cols(), std::cref(Xt), max_leaves);
    kdtree.index->buildIndex();

    Matu knn(k+1, X.cols());
    std::vector<std::size_t> indices(k+1);
    std::vector<impl::Scalar<V>> distances(k+1);
    nanoflann::KNNResultSet<impl::Scalar<V>> results(k+1);

    for(int i = 0; i < X.cols(); ++i)
    {
        results.init(&indices[0], &distances[0]);
        kdtree.index->findNeighbors(results, &X(0, i), searchParams);
        std::copy(indices.begin(), indices.begin() + k + 1, &knn(0, i));
    }

    return knn;
}

} // namespace nlpp
