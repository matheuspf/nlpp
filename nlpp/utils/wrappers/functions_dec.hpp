/** @file
 * 
 *  @brief This file defines some wrapers over functions that calculate gradients or function/gradient values.
 * 
 *  @details The idea is to provide an uniform interface to be used in algorithms that need both function and gradient
 *           to be calculated.
 * 
 *           If an user has, for example, a routine for calculating the values of a function @c f for a given vector @c x,
 *           and also a separate routine for calculating the gradients of @c f, the FunctionGradient class provides a way to
 *           join both routines in a single function/gradient routine.
 * 
 *           Also, if a user has a single routine that calculates both function and gradients value for a function @c f,
 *           FunctionGradient can also provide an interface for calculating function and gradients separatelly:
 *          
 *           @snippet Helpers/Gradient.cpp FunctionGradient snippet
*/

#pragma once

#include "helpers.hpp"



/// Wrap namespace
namespace nlpp::wrap
{

namespace impl
{

using ::nlpp::impl::VecType, ::nlpp::impl::MatType, ::nlpp::impl::Empty;


template <class T>
concept FuncGradTupleType = requires(const T& t)
{
    { std::get<0>(t) } -> std::floating_point;
    { std::get<1>(t) } -> VecType;
};


template <class F, class V>
concept FuncTypeBase = VecType<V> && requires (const F& f, const V& x)
{
    { f(x) } -> std::floating_point;
};


template <class F, class V>
concept GradTypeBase_0 = VecType<V> && requires (const F& f, const V& x)
{
    { f(x) } -> VecType;
};

template <class F, class V>
concept GradTypeBase_1 = VecType<V> && requires (const F& f, const V& x, Plain<V>& g)
{
    { f(x, g) } -> std::same_as<void>;
};

template <class F, class V>
concept GradTypeBase_2 = VecType<V> && requires (const F& f, const V& x, Plain<V>& g, Scalar<V> fx)
{
    { f(x, g, fx) } -> std::same_as<void>;
};

template <class F, class V>
concept GradTypeBase = GradTypeBase_0<F, V> || GradTypeBase_1<F, V> || GradTypeBase_2<F, V>;


template <class F, class V>
concept FuncGradTypeBase_0 = VecType<V> && requires (const F& f, const V& x)
{
    { f(x) } -> FuncGradTupleType;
};

template <class F, class V>
concept FuncGradTypeBase_1 = VecType<V> && requires (const F& f, const V& x, Plain<V>& g)
{
    { f(x, g) } -> std::floating_point;
};

template <class F, class V>
concept FuncGradTypeBase = FuncGradTypeBase_0<F, V> || FuncGradTypeBase_1<F, V>;


template <class F, class V>
concept HessianTypeBase_0 = VecType<V> && requires (const F& f, const V& x)
{
    { f(x) } -> MatType;
};

template <class F, class V>
concept HessianTypeBase_1 = VecType<V> && requires (const F& f, const V& x, Plain2D<V>& h)
{
    { f(x, h) } -> std::same_as<void>;
};

template <class F, class V>
concept HessianTypeBase = HessianTypeBase_0<F, V> || HessianTypeBase_1<F, V>;


template <class F, class V1, class V2 = V1>
concept GradDirTypeBase_0 = VecType<V1> && VecType<V2> && requires (const F& f, const V1& x, const V2& e)
{
    { f(x, e) } -> std::floating_point;
};

template <class F, class V1, class V2 = V1>
concept GradDirTypeBase_1 = VecType<V1> && VecType<V2> && requires (const F& f, const V1& x, const V2& e, Scalar<V1> fx)
{
    { f(x, e, fx) } -> std::floating_point;
};

template <class F, class V1, class V2 = V1>
concept GradDirTypeBase = GradDirTypeBase_0<F, V1, V2> || GradDirTypeBase_1<F, V1, V2>;


template <class F, class V1, class V2 = V1>
concept HessianDirTypeBase = VecType<V1> && VecType<V2> && requires (const F& f, const V1& x, const V2& e)
{
    { f(x, e) } -> VecType;
};



NLPP_FUNCTOR_CONCEPT(FuncType, FuncTypeBase)

NLPP_FUNCTOR_CONCEPT(GradType_0, GradTypeBase_0)
NLPP_FUNCTOR_CONCEPT(GradType_1, GradTypeBase_1)
NLPP_FUNCTOR_CONCEPT(GradType_2, GradTypeBase_2)
NLPP_FUNCTOR_CONCEPT(GradType, GradTypeBase)

NLPP_FUNCTOR_CONCEPT(FuncGradType_0, FuncGradTypeBase_0)
NLPP_FUNCTOR_CONCEPT(FuncGradType_1, FuncGradTypeBase_1)
NLPP_FUNCTOR_CONCEPT(FuncGradType, FuncGradTypeBase)

NLPP_FUNCTOR_CONCEPT(HessianType_0, HessianTypeBase_0)
NLPP_FUNCTOR_CONCEPT(HessianType_1, HessianTypeBase_1)
NLPP_FUNCTOR_CONCEPT(HessianType, HessianTypeBase)

NLPP_FUNCTOR_CONCEPT(GradDirType_0, GradDirTypeBase_0)
NLPP_FUNCTOR_CONCEPT(GradDirType_1, GradDirTypeBase_1)
NLPP_FUNCTOR_CONCEPT(GradDirType, GradDirTypeBase)

NLPP_FUNCTOR_CONCEPT(HessianDirType, HessianDirTypeBase)


template <Conditions Cond, class... Fs>
struct Functions
{
    using TFs = std::tuple<Fs...>;

    enum : bool
    {
        HasFunction = bool(Cond & Conditions::Function),
        HasGradient = bool(Cond & Conditions::Gradient),
        HasHessian  = bool(Cond & Conditions::Hessian),
    };

    Functions(const Fs&... fs) : fs(fs...)
    {
    }


    template <VecType V, bool Enable = HasFunction> requires Enable
    Scalar<V> function (const V& x) const;


    template <VecType V, bool Enable = HasGradient> requires Enable
    Plain<V> gradient (const V& x) const;

    template <VecType V, bool Enable = HasGradient> requires Enable
    void gradient (const V& x, Plain<V>& g) const;

    template <VecType V, bool Enable = HasGradient> requires Enable
    void gradient (const V& x, Plain<V>& g, Scalar<V> fx) const;


    template <VecType V, bool Enable = HasFunction && HasGradient> requires Enable
    std::pair<Scalar<V>, Plain<V>> funcGrad (const V& x) const;

    template <VecType V, bool Enable = HasFunction && HasGradient> requires Enable
    Scalar<V> funcGrad (const V& x, Plain<V>& g) const;


    template <VecType V, bool Enable = HasHessian> requires Enable
    Plain2D<V> hessian (const V& x) const;

    template <VecType V, bool Enable = HasHessian> requires Enable
    void hessian (const V& x, Plain2D<V>& h) const;


    template <VecType V1, VecType V2, bool Enable = HasGradient> requires Enable
    Scalar<V1> gradientDir (const V1& x, const V2& e) const;

    template <VecType V1, VecType V2, bool Enable = HasGradient> requires Enable
    Scalar<V1> gradientDir (const V1& x, const V2& e, Scalar<V1> fx) const;


    template <VecType V1, VecType V2, bool Enable = HasHessian> requires Enable
    Plain<V1> hessianDir (const V1& x, const V2& e) const;


    // template <class V, bool Enable = HasHessian, std::enable_if_t<Enable, int> = 0>
    // Plain2D<V> hessianFromDirectional (const Eigen::MatrixBase<V>& x) const;

    // template <class V, bool Enable = HasHessian, std::enable_if_t<Enable, int> = 0>
    // void hessianFromDirectional (const Eigen::MatrixBase<V>& x, Plain2D<V>& h) const;


    template <VecType V, bool Enable = HasFunction && !HasGradient && !HasHessian> requires Enable
    Scalar<V> operator() (const V& x) const
    {
        return function(x);
    }


    template <VecType V, bool Enable = !HasFunction && HasGradient && !HasHessian> requires Enable
    Plain<V> operator() (const V& x) const
    {
        return gradient(x);
    }

    template <VecType V, bool Enable = !HasFunction && HasGradient && !HasHessian> requires Enable
    void operator() (const V& x, Plain<V>& g) const
    {
        gradient(x, g);
    }

    template <VecType V, bool Enable = !HasFunction && HasGradient && !HasHessian> requires Enable
    void operator() (const V& x, Plain<V>& g, Scalar<V> fx) const
    {
        gradient(x, g, fx);
    }


    template <VecType V, bool Enable = HasFunction && HasGradient> requires Enable
    std::pair<Scalar<V>, Plain<V>> operator() (const V& x) const
    {
        return funcGrad(x);
    }

    template <VecType V, bool Enable = HasFunction && HasGradient> requires Enable
    Scalar<V> operator() (const V& x, Plain<V>& g) const
    {
        return funcGrad(x, g);
    }


    template <VecType V, bool Enable = !HasFunction && !HasGradient && HasHessian> requires Enable
    Plain2D<V> operator() (const V& x) const
    {
        return hessian(x);
    }

    template <VecType V, bool Enable = !HasFunction && !HasGradient && HasHessian> requires Enable
    void operator() (const V& x, Plain2D<V>& h) const
    {
        hessian(x, h);
    }


    TFs fs;
};

} // namespace impl


using ::nlpp::impl::Scalar, ::nlpp::impl::Plain, ::nlpp::impl::Plain2D, ::nlpp::wrap::impl::HasOp;


template <Conditions Cond, class... Fs>
using Functions = impl::Functions<Cond, Fs...>;

template <class... Fs>
using Function = Functions<Conditions::Function, Fs...>;

template <class... Fs>
using Gradient = Functions<Conditions::Gradient, Fs...>;

template <class... Fs>
using FuncGrad = Functions<Conditions::Function | Conditions::Gradient, Fs...>;

template <class... Fs>
using Hessian = Functions<Conditions::Hessian, Fs...>;


template <Conditions Cond, class... Fs>
constexpr Functions<Cond, Fs...> functions (const Fs&... fs)
{
    return Functions<Cond, Fs...>(fs...);
}

template <class... Fs>
constexpr Function<Fs...> function (const Fs&... fs)
{
    return Function<Fs...>(fs...);
}

template <class... Fs>
constexpr Gradient<Fs...> gradient (const Fs&... fs)
{
    return Gradient<Fs...>(fs...);
}

template <class... Fs>
constexpr FuncGrad<Fs...> funcGrad (const Fs&... fs)
{
    return FuncGrad<Fs...>(fs...);
}

template <class... Fs>
constexpr Hessian<Fs...> hessian (const Fs&... fs)
{
    return Hessian<Fs...>(fs...);
}


namespace fd
{

template <Conditions Cond, impl::VecType V = ::nlpp::Vec, class... Fs>
constexpr auto functions (const Fs&... fs)
{
    using TFs = std::tuple<Fs...>;

    constexpr bool addGradient = bool(Cond & Conditions::Gradient) && !impl::hasOp<impl::FuncGradType_Check, TFs, V> && !impl::hasOp<impl::GradType_Check, TFs, V>;
    constexpr bool addHessian  = bool(Cond & Conditions::Hessian)  && !impl::hasOp<impl::HessianType_Check, TFs, V>;

    constexpr int funcId = impl::opId<impl::FuncType_Check, TFs, V>;

    if constexpr(addGradient || addHessian)
    {
        if constexpr(funcId == -1)
            static_assert(::nlpp::impl::always_false<V>, "The functor has no interface for the given requirements");

        const auto& func = std::get<funcId>(std::forward_as_tuple(fs...));
        using Func = std::decay_t<decltype(func)>;

        using GradientFD = ::nlpp::fd::Gradient<Func, ::nlpp::fd::Forward, ::nlpp::fd::SimpleStep<Scalar<V>>>;
        using HessianFD = ::nlpp::fd::Hessian<Func, ::nlpp::fd::Forward, ::nlpp::fd::SimpleStep<Scalar<V>>, Scalar<V>>;

        if constexpr(addGradient && addHessian)
            return ::nlpp::wrap::functions<Cond>(fs..., GradientFD(func), HessianFD(func));

        else if constexpr(addGradient)
            return ::nlpp::wrap::functions<Cond>(fs..., GradientFD(func));

        else
            return ::nlpp::wrap::functions<Cond>(fs..., HessianFD(func));
    }

    else
        return ::nlpp::wrap::functions<Cond>(fs...);
}

template <impl::VecType V = ::nlpp::Vec, class... Fs>
constexpr auto gradient (const Fs&... fs)
{
    return functions<Conditions::Gradient, V>(fs...);
}

template <impl::VecType V = ::nlpp::Vec, class... Fs>
constexpr auto funcGrad (const Fs&... fs)
{
    return functions<Conditions::Function | Conditions::Gradient, V>(fs...);
}

template <impl::VecType V = ::nlpp::Vec, class... Fs>
constexpr auto hessian (const Fs&... fs)
{
    return functions<Conditions::Hessian, V>(fs...);
}

} // namespace fd


namespace poly
{

using ::nlpp::impl::Scalar, ::nlpp::impl::Plain, ::nlpp::impl::Plain2D, ::nlpp::wrap::impl::HasOp;


template <impl::VecType V = ::nlpp::Vec>
using FunctionBase = std::function<Scalar<V>(const Plain<V>&)>;

template <impl::VecType V = ::nlpp::Vec>
using GradientBase = std::function<void(const Plain<V>&, Plain<V>&)>;

template <impl::VecType V = ::nlpp::Vec>
using FuncGradBase = std::function<Scalar<V>(const Plain<V>&, Plain<V>&)>;

template <impl::VecType V = ::nlpp::Vec>
using HessianBase = std::function<void(const Plain<V>&, Plain2D<V>&)>;

template <impl::VecType V = ::nlpp::Vec>
using Functions = ::nlpp::wrap::impl::Functions<Conditions::AllFunctions, FunctionBase<V>, GradientBase<V>, HessianBase<V>>;


template <impl::VecType V = ::nlpp::Vec, class... Fs>
constexpr Functions<V> functions (const Fs&... fs)
{
    return Functions<V>(fs...);
}

// template <impl::VecType V = ::nlpp::Vec, class... Fs>
// constexpr Functions<V> function (const Fs&... fs)
// {
//     return functions<V>(fs...);
// }

// template <impl::VecType V = ::nlpp::Vec, class... Fs>
// constexpr Functions<V> gradient (const Fs&... fs)
// {
//     return functions<V>(fs...);
// }

// template <impl::VecType V = ::nlpp::Vec, class... Fs>
// constexpr Functions<V> funcGrad (const Fs&... fs)
// {
//     return functions<V>(fs...);
// }

// template <impl::VecType V = ::nlpp::Vec, class... Fs>
// constexpr Functions<V> hessian (const Fs&... fs)
// {
//     return functions<V>(fs...);
// }


namespace fd
{

template <Conditions Cond, impl::VecType V = ::nlpp::Vec, class... Fs>
constexpr auto functions (const Fs&... fs)
{
    using TFs = std::tuple<Fs...>;

    FunctionBase<V> function;
    GradientBase<V> gradient;
    HessianBase<V> hessian;

    constexpr bool addGradient = bool(Cond & Conditions::Gradient) && !impl::hasOp<impl::FuncGradType_Check, TFs, V> && !impl::hasOp<impl::GradType_Check, TFs, V>;
    constexpr bool addHessian  = bool(Cond & Conditions::Hessian)  && !impl::hasOp<impl::HessianType_Check, TFs, V>;

    if constexpr(addGradient || addHessian)
    {
        constexpr int funcId = impl::opId<impl::FuncType_Check, TFs, V>;

        if constexpr((addGradient || addHessian) && funcId == -1)
            static_assert(::nlpp::impl::always_false<V>, "The functor has no interface for the given requirements");

        const auto& func = std::get<funcId>(std::forward_as_tuple(fs...));
        using Func = std::decay_t<decltype(func)>;

        using GradientFD = ::nlpp::fd::Gradient<Func, ::nlpp::fd::Forward, ::nlpp::fd::AutoStep>;
        using HessianFD = ::nlpp::fd::Hessian<Func, ::nlpp::fd::Forward, ::nlpp::fd::AutoStep, Scalar<V>>;

        function = [f = ::nlpp::wrap::function(func)](const Plain<V>& x) -> Scalar<V> { return f.function(x); };

        if constexpr(addGradient)
            gradient = [f = ::nlpp::wrap::gradient(GradientFD(func))](const Plain<V>& x, Plain<V>& g) { f.gradient(x, g); };

        if constexpr(addHessian)
            hessian = [f = ::nlpp::wrap::hessian(HessianFD(func))](const Plain<V>& x, Plain2D<V>& h) { f.hessian(x, h); };
    }

    else
    {
        auto wrapper = ::nlpp::wrap::functions<Cond>(fs...);

        function = [f = wrapper](const Plain<V>& x) -> Scalar<V> { return f.function(x); };

        if constexpr(bool(Cond & Conditions::Gradient))
            gradient = [f = wrapper](const Plain<V>& x, Plain<V>& g) { f.gradient(x, g); };

        if constexpr(bool(Cond & Conditions::Hessian))
            hessian = [f = wrapper](const Plain<V>& x, Plain<V>& h) { f.hessian(x, h); };
    }

    return ::nlpp::wrap::poly::functions<V>(function, gradient, hessian);
}

template <impl::VecType V = ::nlpp::Vec, class... Fs>
constexpr auto gradient (const Fs&... fs)
{
    return functions<Conditions::Gradient, V>(fs...);
}

template <impl::VecType V = ::nlpp::Vec, class... Fs>
constexpr auto funcGrad (const Fs&... fs)
{
    return functions<Conditions::Function | Conditions::Gradient, V>(fs...);
}

template <impl::VecType V = ::nlpp::Vec, class... Fs>
constexpr auto hessian (const Fs&... fs)
{
    return functions<Conditions::Hessian, V>(fs...);
}

} // namespace fd


} // namespace poly

} // namespace nlpp::wrap
