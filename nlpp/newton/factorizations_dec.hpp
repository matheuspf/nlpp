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
    SmallIdentity (Float alpha=1e-5) : alpha(alpha)
    {
    }

    template <class V, class U>
    impl::Plain<V> operator () (const Eigen::MatrixBase<V>& grad, U hess) const;

    Float alpha;
};

template <typename Float = types::Float>
struct CholeskyIdentity
{
    CholeskyIdentity (Float beta = 1e-3, Float c = 2.0, Float maxTau = 1e8) : beta(beta), c(c), maxTau(maxTau)
    {
    }

    template <class V, class U>
    impl::Plain<V> operator () (const Eigen::MatrixBase<V>& grad, U hess) const;

    Float beta;
    Float c;
    Float maxTau;
};

template <typename Float = types::Float>
struct CholeskyFactorization
{
    CholeskyFactorization (Float delta = 1e-3) : delta(delta)
    {
    }

    template <class V, class U>
    impl::Plain<V> operator () (const Eigen::MatrixBase<V>& grad, U hess) const;

    Float delta;
};

template <typename Float = types::Float>
struct IndefiniteFactorization
{
    IndefiniteFactorization (Float delta = 1e-2) : delta(delta)
    {
    }

    template <class V, class U>
    impl::Plain<V> operator () (const Eigen::MatrixBase<V>& grad, U hess) const;

    Float delta;
};

} // namespace nlpp::fact