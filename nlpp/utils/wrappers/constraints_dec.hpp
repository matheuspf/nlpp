
#pragma once

#include "helpers.hpp"


namespace nlpp::wrap
{

namespace impl
{

using ::nlpp::impl::VecType, ::nlpp::impl::MatType;


template <class F, class V>
concept ConstraintsFunctorBase = VecType<V> && requires(const F& f, const V& v)
{
    { f(v) } -> VecType;
};

template <class F, typename... Args>
concept ConstraintsFunctorHelper = (ConstraintsFunctorBase<F, Eigen::VectorX<Args>> || ...);

template <class F>
concept ConstraintsFunctor = ConstraintsFunctorHelper<F, float, double, long double>;


template <class F>
concept IneqsFunctor = ConstraintsFunctor<F>;

template <class F>
concept EqsFunctor = ConstraintsFunctor<F>;





// template <class Impl, class V>
// struct IsIneqs : std::bool_constant< IneqsFunctor<Impl, V> > {};

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




template <IneqsFunctor F>
struct IneqsContainer
{
    F ineqs;
};

template <EqsFunctor F>
struct EqsContainer 
{
    F eqs;
};


// template <Conditions Cond>
// constexpr int find_ineqs_index ()
// {
//     if(!(Cond & Conditions::NLInequalities))
//         return -1;
    
//     return 0;
// }

// template <Conditions Cond>
// constexpr int find_eqs_index ()
// {
//     if(!(Cond & Conditions::NLEqualities))
//         return -1;

//     return find_ineqs_index<Cond>() + 1;
// }

// template <Conditions Cond>
// constexpr int find_ineqs_jac_index ()
// {
//     if(!(Cond & Conditions::NLInequalitiesJacobian))
//         return -1;

//     return std::max(find_ineqs_index<Cond>() + 1, 0) + std::max(find_eqs_index<Cond>() + 1, 0);
// }



template <class...>
struct Empty
{
    template <class T=nullptr_t>
    Empty(const std::initializer_list<T>&) {}

    Empty(...) {}
};


template <class BaseClass>

struct BaseOrEmpty : std::conditional_t<std::is_same_v<BaseClass, Empty<>>, Empty<BaseClass>, BaseClass> {};




template <Conditions Cond, class... Fs>
struct Constraints : 
{
    using TFs = std::tuple<Fs...>;

    enum : bool
    {
        HasIneqs        = bool(Cond & Conditions::NLInequalities),
        HasEqs          = bool(Cond & Conditions::NLEqualities),
        HasIneqsEqs     = bool(Cond & )
    };


    template <typename... Args>
    Constraints (Args&&... args) : fs(std::forward<Args>(args)...) {}



    template <VecType V>
    VecType auto ineqs (const V& x) const
    {
        if constexpr(HasIneqs)
            return _ineqs(x);

        else if(constexpr )
            return std::get<0>(_ineqsEqs(x));

        else
            static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }
    
    Ineqs _ineqs;
    Eqs _eqs;
    IneqsEqs _ineqsEqs;








    template <VecType V, bool Enable=HasIneqs>
    requires Enable
    auto ineqs (const V& x) const
    {
        return ineqs_(x);
    }


    template <class V, bool Enable = HasIneqs, std::enable_if_t<Enable, int> = 0>
    auto ineqs_ (const Eigen::MatrixBase<V>& x) const
    {
        if constexpr(HasOp<IsIneqs, V, TFs>)
            return std::get<OpId<IsIneqs, V, TFs>>(fs)(x);

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
        
        else if constexpr(HasIneqs && HasEqs)
            return std::make_pair(ineqs(x), eqs(x));
        
        else if constexpr(HasIneqs)
            return ineqs(x);
        
        else if constexpr(HasEqs)
            return eqs(x);
        
        else
            static_assert(always_false<V>, "The functor has no interface for the given parameter");
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