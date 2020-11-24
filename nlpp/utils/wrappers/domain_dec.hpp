#pragma once

#include "helpers.hpp"

#define NLPP_DEFINE_DOMAIN_VALUE(NAME, ...) \
template <class>    \
struct NAME;       \
\
template <class V>  \
struct NAME<Eigen::MatrixBase<V>>  \
{   \
    const V value;  \
};


namespace nlpp::wrap
{

namespace impl
{

using ::nlpp::impl::Plain, ::nlpp::impl::Plain1D, ::nlpp::impl::Plain2D, ::nlpp::impl::EmptyBase,
      ::nlpp::impl::Scalar, ::nlpp::impl::isVec, ::nlpp::impl::isMat;

template <class V>
struct Start
{
    template <class U, std::enable_if_t<isVec<U>, int> = 0>
    Start (U&& x0) : x0(std::forward<U>(x0)) {}

    Start (int n) : x0(V::Constant(n, 0.0)) {}

    V x0;
};

template <class V>
struct Bounds
{
    template <class U, class W, std::enable_if_t<isVec<U> && isVec<W>, int> = 0>
    Bounds (U&& lb, W&& ub) : lb(std::forward<U>(lb)), ub(std::forward<W>(ub)) {}

    Bounds (int n) : lb(V::Constant(n, -std::numeric_limits<Scalar<V>>::max()/2)), ub(V::Constant(n, std::numeric_limits<Scalar<V>>::max()/2)) {}

    template <class U>
    bool within (const Eigen::MatrixBase<U>& x) const
    {
        return (x.array() >= lb.array()).all() && (x.array() <= ub.array()).all();
    }

    template <class U>
    bool within (const Eigen::MatrixBase<U>& x, int i) const
    {
        return x(i) >= lb(i) && x(i) <= ub(i);
    }

    V lb;
    V ub;
};

template <class V>
struct LinearInequalities
{
    template <class U, class W, std::enable_if_t<isMat<U> && isVec<W>, int> = 0>
    LinearInequalities (U&& A, W&& b) : A(std::forward<U>(A)), b(std::forward<W>(b)) {}

    LinearInequalities (int n) : A(Plain2D<V>::Constant(1, n, 0.0)), b(V::Constant(1, 0.0)) {}

    template <class U>
    auto ineqs (const Eigen::MatrixBase<U>& x) const
    {
        return A * x - b;
    }

    Plain2D<V> A;
    V b;
};

template <class V>
struct LinearEqualities
{
    template <class U, class W, std::enable_if_t<isMat<U> && isVec<W>, int> = 0>
    LinearEqualities (U&& Aeq, W&& beq) : Aeq(std::forward<U>(Aeq)), beq(std::forward<W>(beq)) {}

    LinearEqualities (int n) : Aeq(Plain2D<V>::Constant(1, n, 0.0)), beq(V::Constant(1, 0.0)) {}

    template <class U>
    auto eqs (const Eigen::MatrixBase<U>& x) const
    {
        return Aeq * x - beq;
    }

    Plain2D<V> Aeq;
    V beq;
};

} // namespace impl


using ::nlpp::impl::isVec, ::nlpp::impl::isMat, ::nlpp::impl::Plain1D;

/// V_ must be PLAIN!!!
template <class V_, Conditions Cond>
struct Domain : public std::conditional_t<bool(Cond & Conditions::Start), impl::Start<V_>, ::nlpp::impl::EmptyBase<impl::Start<V_>>>,
                public std::conditional_t<bool(Cond & Conditions::Bounds), impl::Bounds<V_>, ::nlpp::impl::EmptyBase<impl::Bounds<V_>>>,
                public std::conditional_t<bool(Cond & Conditions::LinearInequalities), impl::LinearInequalities<V_>, ::nlpp::impl::EmptyBase<impl::LinearInequalities<V_>>>,
                public std::conditional_t<bool(Cond & Conditions::LinearEqualities), impl::LinearEqualities<V_>, ::nlpp::impl::EmptyBase<impl::LinearEqualities<V_>>>
{
    using V = V_;

    enum : bool
    {
        HasStart = bool(Cond & Conditions::Start),
        HasBounds = bool(Cond & Conditions::Bounds),
        HasLinearInequalities = bool(Cond & Conditions::LinearInequalities),
        HasLinearEqualities = bool(Cond & Conditions::LinearEqualities)
    };

    using Start = std::conditional_t<bool(Cond & Conditions::Start), impl::Start<V>, ::nlpp::impl::EmptyBase<impl::Start<V>>>;
    using Bounds = std::conditional_t<bool(Cond & Conditions::Bounds), impl::Bounds<V>, ::nlpp::impl::EmptyBase<impl::Bounds<V>>>;
    using LinearInequalities = std::conditional_t<bool(Cond & Conditions::LinearInequalities), impl::LinearInequalities<V>, ::nlpp::impl::EmptyBase<impl::LinearInequalities<V>>>;
    using LinearEqualities = std::conditional_t<bool(Cond & Conditions::LinearEqualities), impl::LinearEqualities<V>, ::nlpp::impl::EmptyBase<impl::LinearEqualities<V>>>;


    template <class V1, std::enable_if_t<isVec<V1> && HasStart, int> = 0>
    Domain (V1&& x0) : Start(std::forward<V1>(x0)), Bounds(x0.rows()), LinearInequalities(x0.rows()), LinearEqualities(x0.rows())
    {
    }

    template <class V1, class V2, std::enable_if_t<(isVec<V1> && isVec<V2>) && HasBounds, int> = 0>
    Domain (V1&& lu, V2&& ub) : Start(lu.rows()), Bounds(std::forward<V1>(lu), std::forward<V2>(ub)), LinearInequalities(lu.rows()), LinearEqualities(lu.rows())
    {
    }

    template <class V1, class V2, class V3, std::enable_if_t<(isVec<V1> && isVec<V2> && isVec<V3>) && (HasStart && HasBounds), int> = 0>
    Domain (V1&& x0, V2&& lb, V3&& ub) : Start(std::forward<V1>(x0)), Bounds(std::forward<V1>(lb), std::forward<V2>(ub)), LinearInequalities(x0.rows()), LinearEqualities(x0.rows())
    {
    }

    template <class V1, class V2, std::enable_if_t<(isMat<V1> && isVec<V2>) && HasLinearInequalities, int> = 0>
    Domain (V1&& A, V2&& b) : Start(A.cols()), Bounds(A.cols()), LinearInequalities(std::forward<V1>(A), std::forward<V2>(b)), LinearEqualities(A.cols())
    {
    }

    template <class V1, class V2, std::enable_if_t<(isMat<V1> && isVec<V2>) && HasLinearEqualities, int> = 0>
    Domain (V1&& Aeq, V2&& beq) : Start(Aeq.cols()), Bounds(Aeq.cols()), LinearInequalities(Aeq.cols()), LinearEqualities(std::forward<V1>(Aeq), std::forward<V2>(beq))
    {
    }
};


template <class V>
using StartDomain = Domain<V, Conditions::Start>;

template <class V>
using BoxDomain = Domain<V, Conditions::Bounds>;

template <Conditions Cond, class V, class... Vs>
auto domain (V&& v, Vs&&... vs)
{
    return Domain<::nlpp::impl::Plain<V>, Cond>(std::forward<V>(v), std::forward<Vs>(vs)...);
}

template <class V, class... Vs>
auto startDomain (V&& v, Vs&&... vs)
{
    return domain<Conditions::Start>(std::forward<V>(v), std::forward<Vs>(vs)...);
}




} // namespace nlpp::wrap

