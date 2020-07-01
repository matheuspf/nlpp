#pragma once

#include "helpers.hpp"


namespace nlpp::wrap
{

namespace impl
{

NLPP_MAKE_CALLER(ineqs, false);
NLPP_MAKE_CALLER(eqs, false);
NLPP_MAKE_CALLER(ineqsJac, false);
NLPP_MAKE_CALLER(eqsJac, false);


template <class Impl, class V>
struct IsIneqs : std::bool_constant< isVec<V> && isVec<ineqsType<Impl, Plain<V>>> > {};

template <class Impl, class V>
struct IsEqs : std::bool_constant< isVec<V> && isVec<eqsType<Impl, Plain<V>>> > {};

template <class Impl, class V>
struct IsConstraint : std::bool_constant< isVec<V> && isVec<CallOpType<Impl, Plain<V>>> > {};


template <class Impl, class V>
struct IsIneqsJac : std::bool_constant< isVec<V> && isMat<ineqsJacType<Impl, Plain<V>>> > {};

template <class Impl, class V>
struct IsEqsJac : std::bool_constant< isVec<V> && isMat<eqsJacType<Impl, Plain<V>>> > {};

template <class Impl, class V>
struct IsJacobian : std::bool_constant< isVec<V> && isMat<CallOpType<Impl, Plain<V>>> > {};



template <class... Fs>
struct ConstraintsVisitor
{
    using TFs = std::tuple<Fs...>;

    static_assert(sizeof...(Fs) <= 4, "Invalid number of functors");


    ConstraintsVisitor (const Fs&... fs) : fs(fs...)
    {
    }

    template <class V, bool Enable = HasOp<IsIneqs, V, TFs> || HasOpId<IsConstraint, V, TFs, 0>, std::enable_if_t<Enable, int> = 0>
    auto ineqs (const Eigen::MatrixBase<V>& x) const
    {
        constexpr auto id = HasOp<IsIneqs, V, TFs> ? OpId<IsIneqs, V, TFs> : 0;
        return ineqsCall(std::get<id>(fs), x);
    }

    template <class V, bool Enable = HasOp<IsEqs, V, TFs> || HasOpId<IsConstraint, V, TFs, 1>, std::enable_if_t<Enable, int> = 0>
    auto eqs (const Eigen::MatrixBase<V>& x) const
    {
        constexpr auto id = HasOp<IsEqs, V, TFs> ? OpId<IsEqs, V, TFs> : 1;
        return eqsCall(std::get<OpId<IsEqs, V, TFs>>(fs), x);
    }

    template <class V, bool Enable = (HasOp<IsIneqs, V, TFs> || HasOpId<IsConstraint, V, TFs, 0>) &&
                                     (HasOp<IsEqs, V, TFs> || HasOpId<IsConstraint, V, TFs, 1>), std::enable_if_t<Enable, int> = 0>
    auto operator() (const Eigen::MatrixBase<V>& x) const
    {
        return std::make_pair(ineqs(x), eqs(x));
    }


    template <class V, bool Enable = HasOp<IsIneqsJac, V, TFs> || HasOpId<IsJacobian, V, TFs, 1> || HasOpId<IsJacobian, V, TFs, 2>, std::enable_if_t<Enable, int> = 0>
    auto ineqsJac (const Eigen::MatrixBase<V>& x) const
    {
        constexpr auto id = HasOp<IsIneqsJac, V, TFs> ? OpId<IsIneqsJac, V, TFs> : (HasOpId<IsJacobian, V, TFs, 1> ? 1 : 2);
        return ineqsJacCall(std::get<id>(fs), x);
    }

    template <class V, bool Enable = HasOp<IsEqsJac, V, TFs> || HasOpId<IsJacobian, V, TFs, 3>, std::enable_if_t<Enable, int> = 0>
    auto eqsJac (const Eigen::MatrixBase<V>& x) const
    {
        constexpr auto id = HasOp<IsEqsJac, V, TFs> ? OpId<IsEqsJac, V, TFs> : 3;
        return eqsJacCall(std::get<id>(fs), x);
    }


    TFs fs;
};


template <Conditions Cond, class Impl>
struct Constraints
{
    using TFs = typename Impl::TFs;

    enum : bool
    {
        HasIneqs        = bool(Cond & Conditions::NLInequalities),
        HasEqs          = bool(Cond & Conditions::NLEqualities),
        HasIneqsJac      = bool(Cond & Conditions::NLInequalitiesJacobian),
        HasEqsJac        = bool(Cond & Conditions::NLEqualitiesJacobian),
    };


    template <typename... Args>
    Constraints (Args&&... args) : impl(std::forward<Args>(args)...) {}


    template <class V, bool Enable = HasIneqs, std::enable_if_t<Enable, int> = 0>
    auto ineqs (const Eigen::MatrixBase<V>& x) const
    {
        if constexpr(HasOp<IsIneqs, V, TFs>)
            return impl.ineqs(x);

        else
            static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }

    template <class V, bool Enable = HasEqs, std::enable_if_t<Enable, int> = 0>
    auto eqs (const Eigen::MatrixBase<V>& x) const
    {
        if constexpr(HasOp<IsEqs, V, TFs>)
            return impl.eqs(x);

        else
            static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }

    template <class V, bool Enable = HasIneqs && HasEqs, std::enable_if_t<Enable, int> = 0>
    auto operator() (const Eigen::MatrixBase<V>& x) const
    {
        if constexpr(HasOp<IsIneqs, V, TFs> && HasOp<IsEqs, V, TFs>)
            return impl.operator()(x);

        else if constexpr(HasOp<IsIneqs, V, TFs>)
            return std::make_pair(ineqs(x), V::Constant(0, 0.0));

        else if constexpr(HasOp<IsIneqs, V, TFs>)
            return std::make_pair(V::Constant(0, 0.0), eqs(x));

        else
            static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }


    template <class V, bool Enable = HasIneqsJac, std::enable_if_t<Enable, int> = 0>
    auto ineqsJac (const Eigen::MatrixBase<V>& x) const
    {
        if constexpr(HasOp<IsIneqsJac, V, TFs>)
            return impl.ineqsJac(x);

        else
            static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }

    template <class V, bool Enable = HasEqsJac, std::enable_if_t<Enable, int> = 0>
    auto eqsJac (const Eigen::MatrixBase<V>& x) const
    {
        if constexpr(HasOp<IsEqsJac, V, TFs>)
            return impl.eqsJac(x);

        else
            static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }


    Impl impl;
};

} // namespace impl


template <Conditions Cond, class... Fs>
using Constraints = impl::Constraints<Cond, impl::ConstraintsVisitor<Fs...>>;

template <class... Fs>
using Unconstrained = impl::Constraints<Conditions::Empty, impl::ConstraintsVisitor<Fs...>>;


template <Conditions Cond, class... Fs>
auto constraints (const Fs&... fs)
{
    return Constraints<Cond, Fs...>(fs...);
}


} // namespace nlpp::wrap