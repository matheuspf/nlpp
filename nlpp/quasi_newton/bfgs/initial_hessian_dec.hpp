#pragma once

#include "../../utils/optimizer.hpp"


namespace nlpp
{

template <typename Float = types::Float>
struct BFGSConstant
{
    BFGSConstant (Float alpha = 1e-4) : alpha(alpha) {}

    template <class Function, class V>
    impl::Plain2D<V> operator() (const Function& f, const Eigen::MatrixBase<V>& x0) const;

    Float alpha;
};

template <typename Float = types::Float>
struct BFGSDiagonal
{
    BFGSDiagonal (Float alpha = 1e-4) : alpha(alpha) {}

    template <class Function, class V>
    impl::Plain2D<V> operator() (const Function& f, const Eigen::MatrixBase<V>& x0) const;

    Float alpha;
};

template <typename Float = types::Float>
struct BFGSIdentity
{
    template <class Function, class V>
    impl::Plain2D<V> operator() (const Function& f, const Eigen::MatrixBase<V>& x0) const;
};

} // namespace nlpp
