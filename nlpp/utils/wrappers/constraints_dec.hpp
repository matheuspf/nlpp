#pragma once

#include "helpers.hpp"


namespace nlpp::wrap
{

namespace impl
{

using ::nlpp::impl::VecType, ::nlpp::impl::MatType, ::nlpp::impl::Empty;


template <class T>
concept TupleVecType = requires(const T& t)
{
    { std::get<0>(t) } -> VecType;
    { std::get<1>(t) } -> VecType;
};


template <class F, class V>
concept IneqsTypeBase = VecType<V> && requires(const F& f, const V& x)
{
    { f(x) } -> VecType;
};

template <class F, class V>
concept EqsTypeBase = IneqsTypeBase<F, V>;


template <class F, class V>
concept IneqsEqsTypeBase = VecType<V> && requires(const F& f, const V& x)
{
    { f(x) } -> TupleVecType;
};



NLPP_FUNCTOR_CONCEPT(IneqsType, IneqsTypeBase)
NLPP_FUNCTOR_CONCEPT(EqsType, EqsTypeBase)
NLPP_FUNCTOR_CONCEPT(IneqsEqsType, IneqsEqsTypeBase)




template <Conditions Cond, class... Fs>
struct Constraints
{
    using TFs = std::tuple<Fs...>;

    enum : bool
    {
        HasIneqs        = bool(Cond & Conditions::NLInequalities),
        HasEqs          = bool(Cond & Conditions::NLEqualities),
        // HasIneqsJac     = bool(Cond & Conditions::NLInequalitiesJacobian),
        // HasEqsJac       = bool(Cond & Conditions::NLEqualitiesJacobian),
    };


    template <typename... Args>
    Constraints (Args&&... args) : fs(std::forward<Args>(args)...) {}


    template <VecType V, bool Enable = HasIneqs> requires Enable
    auto ineqs (const V& x) const
    {
        if constexpr(constexpr int id = opId<IneqsType_Check, TFs, V>; id >= 0)
            return std::get<id>(fs)(x);

        else if constexpr(constexpr int id = opId<IneqsEqsType_Check, TFs, V>; id >= 0)
            return std::get<0>(std::get<id>(fs)(x));

        else
            static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }

    template <VecType V, bool Enable = HasEqs> requires Enable
    auto eqs (const V& x) const
    {
        if constexpr(constexpr int id = opId<EqsType_Check, TFs, V>; id >= 0)
            return std::get<id>(fs)(x);

        else if constexpr(constexpr int id = opId<IneqsEqsType_Check, TFs, V>; id >= 0)
            return std::get<1>(std::get<id>(fs)(x));

        else
            static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }

    template <VecType V, bool Enable = HasIneqs && HasEqs> requires Enable
    auto ineqsEqs (const V& x) const
    {
        if constexpr(constexpr int id = opId<IneqsEqsType_Check, TFs, V>; id >= 0)
            return std::get<id>(fs)(x);

        else if constexpr(hasOp<IneqsType_Check, TFs, V> && hasOp<EqsType_Check, TFs, V>)
            return std::make_tuple(ineqs(x), eqs(x));

        else
            static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }


    template <class V, bool Enable = HasIneqs || HasEqs> requires Enable
    auto operator() (const V& x) const
    {
        if constexpr(hasOp<IneqsType_Check, TFs, V> && hasOp<EqsType_Check, TFs, V>)
            return ineqsEqs(x);

        else if constexpr(hasOp<IneqsType_Check, TFs, V>)
            return ineqs(x);

        else if constexpr(hasOp<EqsType_Check, TFs, V>)
            return eqs(x);

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
