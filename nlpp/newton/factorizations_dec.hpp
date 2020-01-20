#pragma once

#include "../helpers/helpers.hpp"


namespace nlpp::fact
{

struct LLT
{
    template <class V, class U>
    impl::Plain<V> operator () (const Eigen::MatrixBase<V>& grad, const Eigen::MatrixBase<U>& hess) const;
};

struct QR
{
    template <class V, class U>
    impl::Plain<V> operator () (const Eigen::MatrixBase<V>& grad, const Eigen::MatrixBase<U>& hess) const;
};

template <typename Float = types::Float>
struct SmallIdentity
{
    SmallIdentity (Float alpha);

    template <class V, class U>
    impl::Plain<V> operator () (const Eigen::MatrixBase<V>& grad, const Eigen::MatrixBase<U>& hess) const;

    Float alpha;
};

template <typename Float = types::Float>
struct CholeskyIdentity
{
    CholeskyIdentity (Float beta, Float c, Float maxTau);

    template <class V, class U>
    impl::Plain<V> operator () (const Eigen::MatrixBase<V>& grad, const Eigen::MatrixBase<U>& hess) const;

    Float beta;
    Float c;
    Float maxTau;
};

template <typename Float = types::Float>
struct CholeskyFactorization
{
    CholeskyFactorization (Float delta);

    template <class V, class U>
    impl::Plain<V> operator () (const Eigen::MatrixBase<V>& grad, const Eigen::MatrixBase<U>& hess) const;

    Float delta;
};

template <typename Float = types::Float>
struct IndefiniteFactorization
{
    IndefiniteFactorization (Float delta);

    template <class V, class U>
    impl::Plain<V> operator () (const Eigen::MatrixBase<V>& grad, const Eigen::MatrixBase<U>& hess) const;

    Float delta;
};

} // namespace nlpp::fact