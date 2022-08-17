
#pragma once

#include "helpers.hpp"


namespace nlpp::wrap
{

namespace impl
{

using ::nlpp::impl::VecType, ::nlpp::impl::MatType, ::nlpp::impl::Empty;


template <class F, class V>
concept IneqsFunctorBase = VecType<V> && requires(const F& f, const V& v)
{
    { f(v) } -> VecType;
};

template <class F, class V>
concept EqsFunctorBase = IneqsFunctorBase<F, V>;

template <class T>
concept TupleVecType = requires(const T& t)
{
    { std::get<0>(t) } -> VecType;
    { std::get<1>(t) } -> VecType;
};

template <class F, class V>
concept IneqsEqsFunctorBase = VecType<V> && requires(const F& f, const V& v)
{
    { f(v) } -> TupleVecType;
};


NLPP_FUNCTOR_CONCEPT(IneqsFunctor, IneqsFunctorBase)
NLPP_FUNCTOR_CONCEPT(EqsFunctor, EqsFunctorBase)
NLPP_FUNCTOR_CONCEPT(IneqsEqsFunctor, IneqsEqsFunctorBase)




template <class Ineqs, class Eqs, class IneqsEqs>
struct Constraints
{
    enum : bool
    {
        HasIneqs        = !std::same_as<Ineqs, Empty>,
        HasEqs          = !std::same_as<Eqs, Empty>,
        HasIneqsEqs     = !std::same_as<IneqsEqs, Empty>
    };


    template <VecType V>
    VecType auto ineqs (const V& x) const
    {
        if constexpr (HasIneqs)
            return _ineqs(x);

        else if constexpr (HasIneqsEqs)
            return std::get<0>(_ineqsEqs(x));

        // else
        //     static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }

    template <VecType V>
    VecType auto eqs (const V& x) const
    {
        if constexpr (HasEqs)
            return _eqs(x);

        else if constexpr (HasIneqsEqs)
            return std::get<1>(_ineqsEqs(x));

        // else
        //     static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }

    template <VecType V>
    auto ineqsEqs (const V& x) const
    {
        if constexpr(HasIneqsEqs)
            return _ineqsEqs(x);
        
        else if constexpr(HasIneqs && HasEqs)
            return std::forward_as_tuple(_ineqs(x), _eqs(x));
        
        else if constexpr(HasIneqs)
            return std::forward_as_tuple(_ineqs(x), Eigen::VectorX<Scalar<V>>{});

        else if constexpr(HasEqs)
            return std::forward_as_tuple(Eigen::VectorX<Scalar<V>>{}, _eqs(x));

        // else
        //     static_assert(always_false<V>, "The functor has no interface for the given parameter");
    }

    template <VecType V>
    auto operator() (const V& x) const
    {
        return ineqsEqs(x);
    }


    Ineqs _ineqs;
    Eqs _eqs;
    IneqsEqs _ineqsEqs;
};

} // namespace impl


using ::nlpp::impl::Empty, impl::IneqsFunctor, impl::EqsFunctor, impl::IneqsEqsFunctor;

template <class Ineqs = Empty, class Eqs = Empty, class IneqsEqs = Empty>
using Constraints = impl::Constraints<Ineqs, Eqs, IneqsEqs>;

auto constraints ()
{
    return Constraints<>{};
}

template <class Ineqs, class Eqs, class IneqsEqs>
auto constraints (const Constraints<Ineqs, Eqs, IneqsEqs>& c)
{
    return c;
}

template <IneqsFunctor Ineqs>
auto constraints (Ineqs&& ineqs, Empty={}, Empty={})
{
    return Constraints<Ineqs, Empty, Empty>{std::forward<Ineqs>(ineqs), {}, {}};
}

template <IneqsEqsFunctor IneqsEqs>
auto constraints (IneqsEqs&& ineqsEqs, Empty={}, Empty={})
{
    return Constraints<Empty, Empty, IneqsEqs>{{}, {}, std::forward<IneqsEqs>(ineqsEqs)};
}

template <IneqsFunctor Ineqs, EqsFunctor Eqs>
auto constraints (Ineqs&& ineqs, Eqs&& eqs, Empty={})
{
    return Constraints<Ineqs, Eqs, Empty>{
        std::forward<Ineqs>(ineqs),
        std::forward<Eqs>(eqs),
        {}
    };
}

template <IneqsFunctor Ineqs, EqsFunctor Eqs, IneqsEqsFunctor IneqsEqs>
auto constraints (Ineqs&& ineqs, Eqs&& eqs, IneqsEqs&& ineqsEqs)
{
    return Constraints<Ineqs, Eqs, IneqsEqs>{
        std::forward<Ineqs>(ineqs),
        std::forward<Eqs>(eqs),
        std::forward<IneqsEqs>(ineqsEqs)
    };
}





} // namespace nlpp::wrap