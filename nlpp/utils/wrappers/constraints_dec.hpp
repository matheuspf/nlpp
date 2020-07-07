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
struct IsIneqsEqs : std::bool_constant< isVec<V> &&
                                        isVec<NthArg<0, CallOpType<Impl, Plain<V>>>> &&
                                        isVec<NthArg<1, CallOpType<Impl, Plain<V>>>> > {};


template <class Impl, class V>
struct IsIneqsJac : std::bool_constant< isVec<V> && isMat<ineqsJacType<Impl, Plain<V>>> > {};

template <class Impl, class V>
struct IsEqsJac : std::bool_constant< isVec<V> && isMat<eqsJacType<Impl, Plain<V>>> > {};

template <class Impl, class V>
struct IsJacobian : std::bool_constant< isVec<V> && isMat<CallOpType<Impl, Plain<V>>> > {};



template <Conditions Cond, class... Fs>
struct Constraints
{
    using TFs = std::tuple<Fs...>;

    enum : bool
    {
        HasIneqs        = bool(Cond & Conditions::NLInequalities),
        HasEqs          = bool(Cond & Conditions::NLEqualities),
        HasIneqsJac     = bool(Cond & Conditions::NLInequalitiesJacobian),
        HasEqsJac       = bool(Cond & Conditions::NLEqualitiesJacobian),
    };


    template <typename... Args>
    Constraints (Args&&... args) : fs(std::forward<Args>(args)...) {}


    template <class V, bool Enable = HasIneqs, std::enable_if_t<Enable, int> = 0>
    auto ineqs (const Eigen::MatrixBase<V>& x) const
    {
        if constexpr(HasOp<IsIneqs, V, TFs>)
            return std::get<OpId<IsIneqs, V, TFs>>(fs).ineqs(x);

        else if constexpr(IsConstraint<NthArg<0, Fs...>, V>::value)
            return std::get<0>(fs)(x);

        else if constexpr(IsIneqsEqs<NthArg<0, Fs...>, V>::value)
            return std::get<0>(std::get<0>(fs)(x));

        else
            static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }

    template <class V, bool Enable = HasEqs, std::enable_if_t<Enable, int> = 0>
    auto eqs (const Eigen::MatrixBase<V>& x) const
    {
        if constexpr(HasOp<IsEqs, V, TFs>)
            return std::get<OpId<IsEqs, V, TFs>>(fs).eqs(x);

        else if constexpr(HasIneqs && IsConstraint<NthArg<1, Fs...>, V>::value)
            return std::get<1>(fs)(x);

        else if constexpr(!HasIneqs && IsConstraint<NthArg<0, Fs...>, V>::value)
            return std::get<0>(fs)(x);

        else if constexpr(IsIneqsEqs<NthArg<0, Fs...>, V>::value)
            return std::get<1>(std::get<0>(fs)(x));

        else
            static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }

    template <class V, bool Enable = HasIneqs && HasEqs, std::enable_if_t<Enable, int> = 0>
    auto operator() (const Eigen::MatrixBase<V>& x) const
    {
        if constexpr(IsIneqsEqs<NthArg<0, Fs...>, V>::value)
            return std::get<0>(fs)(x);

        else
            return std::make_pair(ineqs(x), eqs(x));
    }


    template <class V, bool Enable = HasIneqsJac, std::enable_if_t<Enable, int> = 0>
    auto ineqsJac (const Eigen::MatrixBase<V>& x) const
    {
        if constexpr(HasOp<IsIneqsJac, V, TFs>)
            return std::get<OpId<IsIneqsJac, V, TFs>>(fs).ineqsJac(x);

        else if constexpr(!HasEqs && IsJacobian<NthArg<1, Fs...>, V>::value)
            return std::get<1>(fs)(x);

        else if constexpr(HasEqs && IsJacobian<NthArg<2, Fs...>, V>::value)
            return std::get<2>(fs)(x);

        else
            static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }

    template <class V, bool Enable = HasEqsJac, std::enable_if_t<Enable, int> = 0>
    auto eqsJac (const Eigen::MatrixBase<V>& x) const
    {
        if constexpr(HasOp<IsEqsJac, V, TFs>)
            return std::get<OpId<IsEqsJac, V, TFs>>(fs).eqsJac(x);

        else if constexpr(!HasIneqs && IsJacobian<NthArg<1, Fs...>, V>::value)
            return std::get<1>(fs)(x);

        else if constexpr(HasIneqs && IsJacobian<NthArg<3, Fs...>, V>::value)
            return std::get<3>(fs)(x);

        else
            static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }


    TFs fs;
};

} // namespace impl


template <Conditions Cond, class... Fs>
using Constraints = impl::Constraints<Cond, Fs...>;

template <class... Fs>
using Unconstrained = impl::Constraints<Conditions::Empty, Fs...>;


template <Conditions Cond, class... Fs>
auto constraints (const Fs&... fs)
{
    return Constraints<Cond, Fs...>(fs...);
}


} // namespace nlpp::wrap
