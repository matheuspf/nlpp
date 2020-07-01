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

NLPP_MAKE_CALLER(function, true);
NLPP_MAKE_CALLER(gradient, true);
NLPP_MAKE_CALLER(funcGrad, true);
NLPP_MAKE_CALLER(hessian, true);


template <class Impl, class V>
struct IsFunction : std::bool_constant< isVec<V> && std::is_floating_point_v<functionType<Impl, Plain<V>>> > {};


template <class Impl, class V>
struct IsGradient_0 : std::bool_constant< isVec<V> && std::is_same_v<gradientType<Impl, Plain<V>, Plain<V>&>, void> > {};

template <class Impl, class V>
struct IsGradient_1 : std::bool_constant< isVec<V> && isVec<gradientType<Impl, Plain<V>>> > {};

template <class Impl, class V>
struct IsGradient_2 : std::bool_constant< isVec<V> && std::is_same_v<gradientType<Impl, Plain<V>, Plain<V>&, Scalar<V>>, void> > {};

template <class Impl, class V>
struct IsGradient_3 : std::bool_constant< isVec<V> && std::is_floating_point_v<gradientType<Impl, Plain<V>, Plain<V>>> > {};

template <class Impl, class V>
struct IsGradient_4 : std::bool_constant< isVec<V> && std::is_floating_point_v<gradientType<Impl, Plain<V>, Plain<V>, Scalar<V>>> > {};

template <class Impl, class V>
struct IsGradient : std::bool_constant< IsGradient_0<Impl, V>::value || IsGradient_1<Impl, V>::value ||
                                        IsGradient_2<Impl, V>::value || IsGradient_3<Impl, V>::value || IsGradient_4<Impl, V>::value > {};


template <class Impl, class V>
struct IsFuncGrad_0 : std::bool_constant< isVec<V> && std::is_floating_point_v<funcGradType<Impl, Plain<V>, Plain<V>&, bool>> > {};

template <class Impl, class V>
struct IsFuncGrad_1 : std::bool_constant< isVec<V> && std::is_floating_point_v<funcGradType<Impl, Plain<V>, Plain<V>&>> > {};

template <class Impl, class V>
struct IsFuncGrad_2 : std::bool_constant< isVec<V> && std::is_floating_point_v<NthArg<0, funcGradType<Impl, Plain<V>>>> &&
                                          isVec<NthArg<1, funcGradType<Impl, Plain<V>>>> > {};

template <class Impl, class V>
struct IsFuncGrad : std::bool_constant< IsFuncGrad_0<Impl, V>::value || IsFuncGrad_1<Impl, V>::value || IsFuncGrad_2<Impl, V>::value > {};


template <class Impl, class V>
struct IsHessian_0 : std::bool_constant< isVec<V> && std::is_same_v<hessianType<Impl, Plain<V>, Plain2D<V>&>, void> > {};

template <class Impl, class V>
struct IsHessian_1 : std::bool_constant< isVec<V> && isMat<hessianType<Impl, Plain<V>>> > {};

template <class Impl, class V>
struct IsHessian_2 : std::bool_constant< isVec<V> && isVec<hessianType<Impl, Plain<V>, Plain<V>>> > {};

template <class Impl, class V>
struct IsHessian : std::bool_constant< IsHessian_0<Impl, V>::value || IsHessian_1<Impl, V>::value || IsHessian_2<Impl, V>::value > {};



template <class... Fs>
struct Visitor
{
    using TFs = std::tuple<Fs...>;

    // Visitor() {}

    Visitor(const Fs&... fs) : fs(fs...)
    {
    }

    template <class V, bool Enable = HasOp<IsFunction, V, TFs>, std::enable_if_t<Enable, int> = 0>
    Scalar<V> function (const Eigen::MatrixBase<V>& x) const
    {
        return functionCall(std::get<OpId<IsFunction, V, TFs>>(fs), x);
    }

    template <class V, bool Enable = HasOp<IsGradient_0, V, TFs>, std::enable_if_t<Enable, int> = 0>
    void gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g) const
    {
        gradientCall(std::get<OpId<IsGradient_0, V, TFs>>(fs), x, g);
    }

    template <class V, bool Enable = HasOp<IsGradient_1, V, TFs>, std::enable_if_t<Enable, int> = 0>
    Plain<V> gradient (const Eigen::MatrixBase<V>& x) const
    {
        return gradientCall(std::get<OpId<IsGradient_1, V, TFs>>(fs), x);
    }

    template <class V, bool Enable = HasOp<IsGradient_2, V, TFs>, std::enable_if_t<Enable, int> = 0>
    void gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V> fx) const
    {
        gradientCall(std::get<OpId<IsGradient_2, V, TFs>>(fs), x, g, fx);
    }


    template <class V, bool Enable = HasOp<IsFuncGrad_0, V, TFs> || HasOp<IsFuncGrad_1, V, TFs>, std::enable_if_t<Enable, int> = 0>
    Scalar<V> funcGrad (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad = true) const
    {
        if constexpr(HasOp<IsFuncGrad_0, V, TFs>)
            return funcGradCall(std::get<OpId<IsFuncGrad_0, V, TFs>>(fs), x, g, calcGrad);

        else
            return funcGradCall(std::get<OpId<IsFuncGrad_1, V, TFs>>(fs), x, g);
    }

    template <class V, bool Enable = HasOp<IsFuncGrad_1, V, TFs>, std::enable_if_t<Enable, int> = 0>
    Scalar<V> funcGrad (const Eigen::MatrixBase<V>& x, Plain<V>& g) const
    {
        return funcGradCall(std::get<OpId<IsFuncGrad_1, V, TFs>>(fs), x, g);
    }

    template <class V, bool Enable = HasOp<IsFuncGrad_2, V, TFs>, std::enable_if_t<Enable, int> = 0>
    std::pair<Scalar<V>, Plain<V>> funcGrad (const Eigen::MatrixBase<V>& x) const
    {
        return funcGradCall(std::get<OpId<IsFuncGrad_2, V, TFs>>(fs), x);
    }


    template <class V, bool Enable = HasOp<IsHessian_0, V, TFs>, std::enable_if_t<Enable, int> = 0>
    void hessian (const Eigen::MatrixBase<V>& x, Plain2D<V>& h) const
    {
        hessianCall(std::get<OpId<IsHessian_0, V, TFs>>(fs), x, h);
    }

    template <class V, bool Enable = HasOp<IsHessian_1, V, TFs>, std::enable_if_t<Enable, int> = 0>
    Plain2D<V> hessian (const Eigen::MatrixBase<V>& x) const
    {
        return hessianCall(std::get<OpId<IsHessian_1, V, TFs>>(fs), x);
    }


    template <class V, class U = V, bool Enable = HasOp<IsGradient_3, V, TFs>, std::enable_if_t<Enable, int> = 0>
    Scalar<V> gradientDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e) const
    {
        return gradientCall(std::get<HasOp<IsGradient_3, V, TFs>>(fs), x, e);
    }

    template <class V, class U = V, bool Enable = HasOp<IsGradient_4, V, TFs>, std::enable_if_t<Enable, int> = 0>
    Scalar<V> gradientDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e, Scalar<V> fx) const
    {
        return gradientCall(std::get<HasOp<IsGradient_4, V, TFs>>(fs), x, e, fx);
    }

    template <class V, class U = V, bool Enable = HasOp<IsHessian_2, V, TFs>, std::enable_if_t<Enable, int> = 0>
    Plain<V> hessianDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e) const
    {
        return hessianCall(std::get<OpId<IsHessian_2, V, TFs>>(fs), x, e);
    }


    template <class V, bool Enable = HasOp<IsFuncGrad, V, TFs>, std::enable_if_t<Enable, int> = 0>
    Scalar<V> getFuncGrad (const Eigen::MatrixBase<V>& x) const
    {
        static Plain<V> g;

        if(!HasOp<IsFuncGrad_0, V, TFs> && !HasOp<IsFuncGrad_2, V, TFs>)
            g.resize(x.rows());

        return getFuncGrad(x, g, false);
    }

    template <class V, bool Enable = HasOp<IsFuncGrad, V, TFs>, std::enable_if_t<Enable, int> = 0>
    Scalar<V> getFuncGrad (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad = true) const
    {
        if constexpr(HasOp<IsFuncGrad_0, V, TFs>)
            return funcGradCall(std::get<OpId<IsFuncGrad_0, V, TFs>>(fs), x, g, calcGrad);

        else if constexpr(HasOp<IsFuncGrad_1, V, TFs>)
            return funcGradCall(std::get<OpId<IsFuncGrad_1, V, TFs>>(fs), x, g);

        else if constexpr(HasOp<IsFuncGrad_2, V, TFs>)
        {
            Scalar<V> f;
            std::tie(f, g) = funcGradCall(std::get<OpId<IsFuncGrad_2, V, TFs>>(fs), x);
            return f;
        }
    }

    std::tuple<Fs...> fs;
};


template <Conditions Cond, class Impl>
struct Functions
{
    using TFs = typename Impl::TFs;

    enum : bool
    {
        HasFunction = bool(Cond & Conditions::Function),
        HasGradient = bool(Cond & Conditions::Gradient),
        HasHessian  = bool(Cond & Conditions::Hessian),
    };


    template <typename... Args>
    Functions (Args&&... args) : impl(std::forward<Args>(args)...) {}


    template <class V, bool Enable = HasFunction, std::enable_if_t<Enable, int> = 0>
    Scalar<V> function (const Eigen::MatrixBase<V>& x) const;

    template <class V, bool Enable = (HasFunction && !HasGradient), std::enable_if_t<Enable, int> = 0>
    Scalar<V> operator() (const Eigen::MatrixBase<V>& x) const
    {
        return function(x);
    }


    template <class V, bool Enable = HasGradient, std::enable_if_t<Enable, int> = 0>
    void gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g) const;

    template <class V, bool Enable = (HasGradient && !HasFunction), std::enable_if_t<Enable, int> = 0>
    void operator()(const Eigen::MatrixBase<V>& x, Plain<V>& g) const
    {
        gradient(x, g);
    }

    template <class V, bool Enable = HasGradient, std::enable_if_t<Enable, int> = 0>
    Plain<V> gradient (const Eigen::MatrixBase<V>& x) const;

    template <class V, bool Enable = (HasGradient && !HasFunction), std::enable_if_t<Enable, int> = 0>
    Plain<V> operator() (const Eigen::MatrixBase<V>& x) const
    {
        return gradient(x);
    }

    template <class V, bool Enable = HasGradient, std::enable_if_t<Enable, int> = 0>
    void gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V> fx) const;

    template <class V, bool Enable = (HasGradient && !HasFunction), std::enable_if_t<Enable, int> = 0>
    void operator() (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V> fx) const
    {
        return gradient(x, g, fx);
    }


    template <class V, bool Enable = HasGradient, std::enable_if_t<Enable, int> = 0>
    Scalar<V> funcGrad (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad = true) const;

    template <class V, bool Enable = (HasGradient && HasFunction), std::enable_if_t<Enable, int> = 0>
    Scalar<V> operator() (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad = true) const
    {
        return funcGrad(x, g, calcGrad);
    }

    template <class V, bool Enable = HasGradient, std::enable_if_t<Enable, int> = 0>
    std::pair<Scalar<V>, Plain<V>> funcGrad (const Eigen::MatrixBase<V>& x) const;

    template <class V, bool Enable = (HasGradient && HasFunction), std::enable_if_t<Enable, int> = 0>
    std::pair<Scalar<V>, Plain<V>> operator() (const Eigen::MatrixBase<V>& x) const
    {
        return funcGrad(x);
    }


    template <class V, bool Enable = HasHessian, std::enable_if_t<Enable, int> = 0>
    void hessian (const Eigen::MatrixBase<V>& x, Plain2D<V>& h) const;

    template <class V, bool Enable = (HasHessian && !HasFunction && !HasGradient), std::enable_if_t<Enable, int> = 0>
    void operator() (const Eigen::MatrixBase<V>& x, Plain2D<V>& h) const
    {
        hessian(x, h);
    }

    template <class V, bool Enable = HasHessian, std::enable_if_t<Enable, int> = 0>
    Plain2D<V> hessian (const Eigen::MatrixBase<V>& x) const;

    template <class V, bool Enable = (HasHessian && !HasFunction && !HasGradient), std::enable_if_t<Enable, int> = 0>
    Plain2D<V> operator() (const Eigen::MatrixBase<V>& x) const
    {
        return hessian(x);
    }


    template <class V, bool Enable = HasHessian, std::enable_if_t<Enable, int> = 0>
    Plain2D<V> hessianFromDirectional (const Eigen::MatrixBase<V>& x) const;

    template <class V, bool Enable = HasHessian, std::enable_if_t<Enable, int> = 0>
    void hessianFromDirectional (const Eigen::MatrixBase<V>& x, Plain2D<V>& h) const;


    template <class V, class U = V, bool Enable = HasGradient, std::enable_if_t<Enable, int> = 0>
    Scalar<V> gradientDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e) const;

    template <class V, class U = V, bool Enable = HasGradient, std::enable_if_t<Enable, int> = 0>
    Scalar<V> gradientDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e, Scalar<V> fx) const;

    template <class V, class U = V, bool Enable = HasHessian, std::enable_if_t<Enable, int> = 0>
    Plain<V> hessianDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e) const;


    Impl impl;
};

} // namespace impl


using ::nlpp::impl::Scalar, ::nlpp::impl::Plain, ::nlpp::impl::Plain2D, ::nlpp::wrap::impl::HasOp;


// template <Conditions Cond, class... Fs>
// using Functions = std::conditional_t<handy::IsSpecialization<::nlpp::impl::FirstArg<Fs...>, impl::Functions>::value, ::nlpp::impl::FirstArg<Fs...>, impl::Functions<Cond, impl::Visitor<Fs...>>>;

template <Conditions Cond, class... Fs>
using Functions = impl::Functions<Cond, impl::Visitor<Fs...>>;

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

template <Conditions Cond, class V = ::nlpp::Vec, class... Fs>
constexpr auto functions (const Fs&... fs)
{
    using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
    using TFs = typename Impl::TFs;

    constexpr bool addGradient = bool(Cond & Conditions::Gradient) && !impl::HasOp<impl::IsFuncGrad, V, TFs> && !impl::HasOp<impl::IsGradient, V, TFs>;
    constexpr bool addHessian  = bool(Cond & Conditions::Hessian)  && !impl::HasOp<impl::IsHessian, V, TFs>;

    constexpr int idFunction = impl::OpId<impl::IsFunction, V, TFs>;

    if constexpr(!addGradient && !addHessian)
        return ::nlpp::wrap::functions(fs...);

    if constexpr((addGradient || addHessian) && idFunction == -1)
        static_assert(::nlpp::impl::always_false<V>, "The functor has no interface for the given requirements");


    const auto& func = std::get<idFunction>(std::forward_as_tuple(fs...));
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

template <class V = ::nlpp::Vec, class... Fs>
constexpr auto gradient (const Fs&... fs)
{
    return functions<Conditions::Gradient, V>(fs...);
}

template <class V = ::nlpp::Vec, class... Fs>
constexpr auto funcGrad (const Fs&... fs)
{
    return functions<Conditions::Function | Conditions::Gradient, V>(fs...);
}

template <class V = ::nlpp::Vec, class... Fs>
constexpr auto hessian (const Fs&... fs)
{
    return functions<Conditions::Hessian, V>(fs...);
}

} // namespace fd


namespace poly
{

using ::nlpp::impl::Scalar, ::nlpp::impl::Plain, ::nlpp::impl::Plain2D, ::nlpp::wrap::impl::HasOp;


template <class V = ::nlpp::Vec>
using FunctionBase = std::function<Scalar<V>(const Plain<V>&)>;

template <class V = ::nlpp::Vec>
using GradientBase = std::function<void(const Plain<V>&, Plain<V>&)>;

template <class V = ::nlpp::Vec>
using FuncGradBase = std::function<Scalar<V>(const Plain<V>&, Plain<V>&, bool)>;

template <class V = ::nlpp::Vec>
using HessianBase = std::function<void(const Plain<V>&, Plain2D<V>&)>;


template <class V = ::nlpp::Vec>
using Visitor = ::nlpp::wrap::impl::Visitor<FunctionBase<V>, GradientBase<V>, HessianBase<V>>;
// using Visitor = ::nlpp::wrap::impl::Visitor<Function<V>, Gradient<V>, FunctionGradient<V>, Hessian<V>>;

template <class V = ::nlpp::Vec>
using Functions = ::nlpp::wrap::Functions<Conditions::AllFunctions, Visitor<V>>;


template <class V = ::nlpp::Vec, class... Fs>
constexpr Functions<V> functions (const Fs&... fs)
{
    return Functions<V>(fs...);
}

template <class V = ::nlpp::Vec, class... Fs>
constexpr Functions<V> function (const Fs&... fs)
{
    return functions<V>(fs...);
}

template <class V = ::nlpp::Vec, class... Fs>
constexpr Functions<V> gradient (const Fs&... fs)
{
    return functions<V>(fs...);
}

template <class V = ::nlpp::Vec, class... Fs>
constexpr Functions<V> funcGrad (const Fs&... fs)
{
    return functions<V>(fs...);
}

template <class V = ::nlpp::Vec, class... Fs>
constexpr Functions<V> hessian (const Fs&... fs)
{
    return functions<V>(fs...);
}


namespace fd
{

template <Conditions Cond, class V = ::nlpp::Vec, class... Fs>
constexpr auto functions (const Fs&... fs)
{
    using TFs = std::tuple<Fs...>;

    Function<V> function;
    Gradient<V> gradient;
    Hessian<V> hessian;

    constexpr bool addGradient = bool(Cond & Conditions::Gradient) && !impl::HasOp<impl::IsFuncGrad, V, TFs> && !impl::HasOp<impl::IsGradient, V, TFs>;
    constexpr bool addHessian  = bool(Cond & Conditions::Hessian)  && !impl::HasOp<impl::IsHessian, V, TFs>;

    if constexpr(addGradient || addHessian)
    {
        constexpr int idFunction = impl::OpId<impl::IsFunction, V, TFs>;

        if constexpr((addGradient || addHessian) && idFunction == -1)
            static_assert(::nlpp::impl::always_false<V>, "The functor has no interface for the given requirements");

        const auto& func = std::get<idFunction>(std::forward_as_tuple(fs...));
        using Func = std::decay_t<decltype(func)>;

        using GradientFD = ::nlpp::fd::Gradient<Func, ::nlpp::fd::Forward, ::nlpp::fd::AutoStep>;
        using HessianFD = ::nlpp::fd::Hessian<Func, ::nlpp::fd::Forward, ::nlpp::fd::AutoStep, Scalar<V>>;

        function = [impl = ::nlpp::wrap::impl::Visitor<Func>(func)](const Plain<V>& x) -> Scalar<V> { return impl.function(x); };

        if constexpr(addGradient)
            gradient = [impl = ::nlpp::wrap::impl::Visitor<GradientFD>(GradientFD(func))](const Plain<V>& x, Plain<V>& g) { impl.gradient(x, g); };

        if constexpr(addHessian)
            hessian = [impl = ::nlpp::wrap::impl::Visitor<HessianFD>(HessianFD(func))](const Plain<V>& x, Plain2D<V>& h) { impl.hessian(x, h); };
    }

    else
    {
        function = [impl = ::nlpp::wrap::impl::Visitor<Fs...>(fs...)](const Plain<V>& x) -> Scalar<V> { return impl.function(x); };

        if constexpr(bool(Cond & Conditions::Gradient))
            gradient = [impl = ::nlpp::wrap::impl::Visitor<Fs...>(fs...)](const Plain<V>& x, Plain<V>& g) { impl.gradient(x, g); };

        if constexpr(bool(Cond & Conditions::Hessian))
            hessian = [impl = ::nlpp::wrap::impl::Visitor<Fs...>(fs...)](const Plain<V>& x, Plain<V>& h) { impl.hessian(x, h); };
    }

    return ::nlpp::wrap::poly::functions<V>(function, gradient, hessian);
}

template <class V = ::nlpp::Vec, class... Fs>
constexpr auto gradient (const Fs&... fs)
{
    return functions<Conditions::Gradient, V>(fs...);
}

template <class V = ::nlpp::Vec, class... Fs>
constexpr auto funcGrad (const Fs&... fs)
{
    return functions<Conditions::Function | Conditions::Gradient, V>(fs...);
}

template <class V = ::nlpp::Vec, class... Fs>
constexpr auto hessian (const Fs&... fs)
{
    return functions<Conditions::Hessian, V>(fs...);
}

} // namespace fd


} // namespace poly

} // namespace nlpp::wrap
