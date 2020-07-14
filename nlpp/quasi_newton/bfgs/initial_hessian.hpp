#pragma once

#include "initial_hessian_dec.hpp"

namespace nlpp
{

template <typename Float>
template <class Function, class V>
impl::Plain2D<V> BFGSDiagonal<Float>::operator() (const Function& f, const Eigen::MatrixBase<V>& x) const
{
    auto hess = impl::Plain2D<V>::Constant(x.rows(), x.rows(), 0.0);

    hess.diagonal() = (2*h) / (f.gradient((x.array() + h).matrix()) - f.gradient((x.array() - h).matrix())).array();
    
    return hess;
}

template <typename Float>
template <class Function, class V>
impl::Plain2D<V> BFGSConstant<Float>::operator() (const Function& f, const Eigen::MatrixBase<V>& x0) const
{
    auto g0 = f.gradient(x0);
    auto x1 = x0 - alpha * g0;
    auto g1 = f.gradient(x1);

    auto s = x1 - x0;
    auto y = g1 - g0;

    auto hess = (y.dot(s) / y.dot(y)) * impl::Plain2D<Derived>::Identity(x0.rows(), x0.rows());

    return hess;
}

template <typename Float>
template <class Function, class V>
impl::Plain2D<V> BFGSIdentity<Float>::operator() (const Function& f, const Eigen::MatrixBase<V>& x) const
{
    return impl::Plain2D<Derived>::Identity(x.rows(), x.rows());
}

} // namespace nlpp
