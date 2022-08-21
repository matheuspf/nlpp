#pragma once

#include "helpers.hpp"


namespace nlpp::wrap
{

namespace impl
{

using ::nlpp::impl::VecType, ::nlpp::impl::MatType, ::nlpp::impl::Plain, ::nlpp::impl::Plain1D,
      ::nlpp::impl::Empty, ::nlpp::impl::TypeOrEmpty;


template <VecType V>
struct Start
{
    template <VecType U>
    Start (U&& x0) : x0(std::forward<U>(x0)) {}

    Start (int n) : x0(V::Constant(n, 0.0)) {}

    V x0;
};

template <VecType V>
struct Bounds
{
    template <VecType U, VecType W>
    Bounds (U&& lb, W&& ub) : lb(std::forward<U>(lb)), ub(std::forward<W>(ub)) {}

    Bounds (int n) : lb(V::Constant(n, -std::numeric_limits<Scalar<V>>::max()/2)), ub(V::Constant(n, std::numeric_limits<Scalar<V>>::max()/2)) {}

    template <VecType U>
    bool within (const Eigen::MatrixBase<U>& x) const
    {
        return (x.array() >= lb.array()).all() && (x.array() <= ub.array()).all();
    }

    template <VecType U>
    bool within (const Eigen::MatrixBase<U>& x, int i) const
    {
        return x(i) >= lb(i) && x(i) <= ub(i);
    }

    V lb;
    V ub;
};

template <VecType V>
struct LinearInequalities
{
    template <MatType U, VecType W, std::enable_if_t<isMat<U> && isVec<W>, int> = 0>
    LinearInequalities (U&& A, W&& b) : A(std::forward<U>(A)), b(std::forward<W>(b)) {}

    LinearInequalities (int n) : A(Plain2D<V>::Constant(1, n, 0.0)), b(V::Constant(1, 0.0)) {}

    template <VecType U>
    auto ineqs (const Eigen::MatrixBase<U>& x) const
    {
        return A * x - b;
    }

    Plain2D<V> A;
    V b;
};

template <VecType V>
struct LinearEqualities
{
    template <MatType U, VecType W, std::enable_if_t<isMat<U> && isVec<W>, int> = 0>
    LinearEqualities (U&& Aeq, W&& beq) : Aeq(std::forward<U>(Aeq)), beq(std::forward<W>(beq)) {}

    LinearEqualities (int n) : Aeq(Plain2D<V>::Constant(1, n, 0.0)), beq(V::Constant(1, 0.0)) {}

    template <VecType U>
    auto eqs (const Eigen::MatrixBase<U>& x) const
    {
        return Aeq * x - beq;
    }

    Plain2D<V> Aeq;
    V beq;
};

} // namespace impl


using ::nlpp::impl::VecType, ::nlpp::impl::MatType, ::nlpp::impl::Plain, ::nlpp::impl::TypeOrEmptyBase;



template <VecType V_, Conditions Cond>
struct Domain : public TypeOrEmptyBase<impl::Start<V_>, bool(Cond & Conditions::Start)>,
                public TypeOrEmptyBase<impl::Bounds<V_>, bool(Cond & Conditions::Bounds)>,
                public TypeOrEmptyBase<impl::LinearInequalities<V_>, bool(Cond & Conditions::LinearInequalities)>,
                public TypeOrEmptyBase<impl::LinearEqualities<V_>, bool(Cond & Conditions::LinearEqualities)>
{
    using V = V_;

    enum : bool
    {
        HasStart = bool(Cond & Conditions::Start),
        HasBounds = bool(Cond & Conditions::Bounds),
        HasLinearInequalities = bool(Cond & Conditions::LinearInequalities),
        HasLinearEqualities = bool(Cond & Conditions::LinearEqualities)
    };

    using Start = TypeOrEmptyBase<impl::Start<V_>, HasStart>;
    using Bounds = TypeOrEmptyBase<impl::Bounds<V_>, HasBounds>;
    using LinearInequalities = TypeOrEmptyBase<impl::LinearInequalities<V_>, HasLinearInequalities>;
    using LinearEqualities = TypeOrEmptyBase<impl::LinearEqualities<V_>, HasLinearEqualities>;


    template <VecType U, bool Enable = HasStart>
    requires Enable
    Domain (U&& start) : Start(std::forward<U>(start)), Bounds(start.rows()),
                         LinearInequalities(start.rows()), LinearEqualities(start.rows())
    {
    }

    template <VecType U, VecType W, bool Enable = HasBounds>
    requires Enable
    Domain (U&& lb, W&& ub) : Start(lb.rows()), Bounds(std::forward<U>(lb), std::forward<W>(ub)),
                              LinearInequalities(lb.rows()), LinearEqualities(lb.rows())
    {
    }

    template <VecType U, VecType W, VecType Z, bool Enable = HasStart && HasBounds>
    requires Enable
    Domain (U&& start, W&& lb, Z&& ub) : Start(std::forward<U>(start)), Bounds(std::forward<W>(lb), std::forward<Z>(ub)),
                                         LinearInequalities(start.rows()), LinearEqualities(start.rows())
    {
    }


    template <MatType U, VecType W, bool Enable = HasLinearInequalities>
    requires Enable
    Domain (U&& A, W&& b) : Start(b.rows()), Bounds(b.rows()),
                            LinearInequalities(std::forward<U>(A), std::forward<W>(b)),
                            LinearEqualities(b.rows())
    {
    }

    template <MatType U, VecType W, bool Enable = HasLinearEqualities && !HasLinearInequalities, int=0>
    requires Enable
    Domain (U&& Aeq, W&& beq) : Start(beq.rows()), Bounds(beq.rows()), LinearInequalities(beq.rows()), 
                                LinearEqualities(std::forward<U>(Aeq), std::forward<W>(beq))
    {
    }
};


template <VecType V>
using StartDomain = Domain<V, Conditions::Start>;

template <VecType V>
using BoxDomain = Domain<V, Conditions::Bounds>;

template <Conditions Cond, VecType V, VecType... Vs>
auto domain (V&& v, Vs&&... vs)
{
    return Domain<::nlpp::impl::Plain<V>, Cond>(std::forward<V>(v), std::forward<Vs>(vs)...);
}

template <VecType V, VecType... Vs>
auto startDomain (V&& v, Vs&&... vs)
{
    return domain<Conditions::Start>(std::forward<V>(v), std::forward<Vs>(vs)...);
}




} // namespace nlpp::wrap



