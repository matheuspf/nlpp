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

#include "helpers/helpers_dec.hpp"
#include "utils/stop.hpp"
#include "utils/output.hpp"

#define NLPP_MAKE_CALLER(NAME) \
\
template <class T, class... Args> \
using NLPP_CONCAT(NAME, Invoke) = decltype(std::declval<T>().NAME(std::declval<Args>()...));   \
\
template <class Impl, class... Args> \
auto NLPP_CONCAT(NAME, Call) (const Impl& impl, Args&&... args) \
{ \
    if constexpr(is_detected_v<NLPP_CONCAT(NAME, Invoke), Impl, Args...>) \
        return impl.NAME(std::forward<Args>(args)...); \
\
    else if constexpr(is_detected_v<std::invoke_result_t, Impl, Args...>) \
        return impl(std::forward<Args>(args)...); \
\
    else \
        return ::nlpp::impl::nonesuch{}; \
} \
\
template <class Impl, class... Args> \
using NLPP_CONCAT(NAME, Type) = decltype(NLPP_CONCAT(NAME, Call)(std::declval<Impl>(), std::declval<Args>()...));



/// Wrap namespace
namespace nlpp::wrap
{


/** @defgroup GradientBaseGroup Gradient Base
    @copydoc Helpers/Gradient.h
*/
//@{

/** @name
 *  @brief Check if the class @c T is a function, gradient or function/gradient functor, taking parameters of type @c Vec
 * 
 *  @tparam T The class to check
 *  @tparam Vec The type of vector that @c T takes as argument
*/
//@{
namespace impl
{

using ::nlpp::impl::Scalar, ::nlpp::impl::Plain, ::nlpp::impl::Plain2D,
      ::nlpp::impl::isMat, ::nlpp::impl::isVec, ::nlpp::impl::detected_t,
      ::nlpp::impl::is_detected_v, ::nlpp::impl::always_false, ::nlpp::impl::NthArg;


template <template <class, class> class Check, class V, class TFs, class Idx>
struct OpIdImpl;

template <template <class, class> class Check, class V, class... Fs, std::size_t... Is>
struct OpIdImpl<Check, V, std::tuple<Fs...>, std::index_sequence<Is...>>
{
    enum { value = (int(Check<Fs, V>::value * int(Is + 1)) + ...) - 1 };
};

template <template <class, class> class Check, class V, class TFs>
static constexpr int OpId = OpIdImpl<Check, V, TFs, std::make_index_sequence<std::tuple_size_v<TFs>>>::value;

template <template <class, class> class Check, class V, class TFs>
static constexpr bool HasOp = OpId<Check, V, TFs> >= 0;


template <class T, class... Args>
using OperatorType = detected_t<std::invoke_result_t, T, Args...>;


NLPP_MAKE_CALLER(function);
NLPP_MAKE_CALLER(gradient);
NLPP_MAKE_CALLER(funcGrad);
NLPP_MAKE_CALLER(hessian);


template <class Impl, class V>
struct IsFunction : std::bool_constant< isVec<V> && std::is_floating_point_v<functionType<Impl, Plain<V>>> > {};


template <class Impl, class V>
struct IsGradient_0 : std::bool_constant< isVec<V> && std::is_same_v<gradientType<Impl, Plain<V>, Plain<V>&>, void> > {};

template <class Impl, class V>
struct IsGradient_1 : std::bool_constant< isVec<V> && isVec<gradientType<Impl, Plain<V>>> > {};

template <class Impl, class V>
struct IsGradient_2 : std::bool_constant< isVec<V> && std::is_same_v<gradientType<Impl, Plain<V>, Plain<V>&, Scalar<V>>, void> > {};

template <class Impl, class V>
struct IsGradient : std::bool_constant< IsGradient_0<Impl, V>::value || IsGradient_1<Impl, V>::value || IsGradient_2<Impl, V>::value > {};

template <class Impl, class V>
struct IsDirectional : std::bool_constant< isVec<V> && std::is_floating_point_v<gradientType<Impl, Plain<V>, Plain<V>>> > {};


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
struct Visitor;


template <class>
struct VisitorTraits;

template <class... Fs>
struct VisitorTraits<Visitor<Fs...>>
{
    using TFs = typename Visitor<Fs...>::TFs;
};

template <class... Fs>
struct VisitorTraits<std::reference_wrapper<const Visitor<Fs...>>>
{
    using TFs = typename Visitor<Fs...>::TFs;
};



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

    template <class V, bool Enable = HasOp<IsDirectional, V, TFs>, std::enable_if_t<Enable, int> = 0>
    Scalar<V> gradient (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e) const
    {
        return gradientCall(std::get<HasOp<IsDirectional, V, TFs>>(fs), x, e);
    }


    template <class V, bool Enable = HasOp<IsFuncGrad_0, V, TFs> || HasOp<IsFuncGrad_1, V, TFs>, std::enable_if_t<Enable, int> = 0>
    Scalar<V> funcGrad (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad =true) const
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

    template <class V, bool Enable = HasOp<IsHessian_2, V, TFs>, std::enable_if_t<Enable, int> = 0>
    Plain<V> hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e) const
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

/** @brief Function wrapping for user uniform defined function calculation
 * 
*/
template <class Impl>
struct Function
{
    using TFs = typename std::decay_t<Impl>::TFs;


    template <typename... Args>
    Function (Args&&... args) : impl(std::forward<Args>(args)...) {}


    template <class V>
    Scalar<V> function (const Eigen::MatrixBase<V>& x) const;

    template <class V>
    Scalar<V> operator () (const Eigen::MatrixBase<V>& x) const
    {
        return function(x);
    }

    /// Necessary to hide a lambda operator matching the exact arguments
    // template <typename T, int R, int C>
    // Scalar<V> operator () (const Eigen::Matrix<T, R, C>& x);

    Impl impl;
};


template <class Impl>
struct Gradient
{
    using TFs = typename std::decay_t<Impl>::TFs;

    template <typename... Args>
    Gradient (Args&&... args) : impl(std::forward<Args>(args)...) {}


    template <class V>
    void gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g) const;

    template <class V>
    Plain<V> gradient (const Eigen::MatrixBase<V>& x) const;

    template <class V>
    void gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V> fx) const;

    template <class V>
    Scalar<V> gradient (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e) const;


    template <class V>
    void operator() (const Eigen::MatrixBase<V>& x, Plain<V>& g) const
    {
        return gradient(x, g);
    }

    template <class V>
    Plain<V> operator() (const Eigen::MatrixBase<V>& x) const
    {
        return gradient(x);
    }

    template <class V>
    void operator() (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V> fx) const
    {
        return gradient(x, g, fx);
    }

    template <class V>
    Scalar<V> operator() (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e) const
    {
        return gradient(x, e);
    }


    Impl impl;
};


/// Forward declaration

/** @brief Specialization for when both function and gradients are given separatelly
 * 
 *  @tparam Func A functor having <tt>Scalar Func::operator()(const Vec&)</tt> defined
 *  @tparam Func A functor having either <tt>Vec Func::operator()(const Vec&)</tt> or 
 *          <tt>void Func::operator()(const Vec&, Vec&)</tt> defined.
 * 
 *  @note This class inherits both the @c Func and @c Grad templates, wrapping the @c Grad 
 *        into @c Gradient<Grad> first
*/
template <class Impl>
struct FunctionGradient
{
    using TFs = typename std::decay_t<Impl>::TFs;

    template <typename... Args>
    FunctionGradient (Args&&... args) : impl(std::forward<Args>(args)...), func(impl), grad(impl) {}

    template <class V>
    Scalar<V> funcGrad (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad = true) const;

    template <class V>
    std::pair<Scalar<V>, Plain<V>> funcGrad (const Eigen::MatrixBase<V>& x) const;

    template <class V>
    Scalar<V> operator() (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad = true) const
    {
        return funcGrad(x, g, calcGrad);
    }

    template <class V>
    std::pair<Scalar<V>, Plain<V>> operator() (const Eigen::MatrixBase<V>& x) const
    {
        return funcGrad(x);
    }


    template <typename... Args>
    auto function (Args&&... args) const
    {
        return func.function(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto gradient (Args&&... args) const
    {
        return grad.gradient(std::forward<Args>(args)...);
    }


    Impl impl;

    Function<const Impl&> func;
    Gradient<const Impl&> grad;
};


template <class Impl>
struct Hessian : public Impl
{
    using TFs = typename std::decay_t<Impl>::TFs;

    template <typename... Args>
    Hessian (Args&&... args) : impl(std::forward<Args>(args)...) {}


    template <class V>
    void hessian (const Eigen::MatrixBase<V>& x, Plain2D<V>& h) const;

    template <class V>
    Plain2D<V> hessian (const Eigen::MatrixBase<V>& x) const;

    template <class V>
    Plain<V> hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e) const;


    template <class V>
    void operator() (const Eigen::MatrixBase<V>& x, Plain2D<V>& h) const
    {
        hessian(x, h);
    }

    template <class V>
    Plain2D<V> operator() (const Eigen::MatrixBase<V>& x) const
    {
        return hessian(x);
    }

    template <class V>
    Plain<V> operator() (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e) const
    {
        return hessian(x, e);
    }


    template <class V>
    Plain2D<V> hessianFromDirectional (const Eigen::MatrixBase<V>& x) const;

    template <class V>
    void hessianFromDirectional (const Eigen::MatrixBase<V>& x, Plain2D<V>& h) const;


    Impl impl;
};


} // namespace impl


template <class F>
using Function = std::conditional_t<handy::IsSpecialization<F, impl::Function>::value, F, impl::Function<impl::Visitor<F>>>;

template <class... Fs>
using Gradient = std::conditional_t<handy::IsSpecialization<::nlpp::impl::FirstArg<Fs...>, impl::Gradient>::value, ::nlpp::impl::FirstArg<Fs...>, impl::Gradient<impl::Visitor<Fs...>>>;

template <class... Fs>
using FunctionGradient = std::conditional_t<handy::IsSpecialization<::nlpp::impl::FirstArg<Fs...>, impl::FunctionGradient>::value, ::nlpp::impl::FirstArg<Fs...>, impl::FunctionGradient<impl::Visitor<Fs...>>>;

template <class... Fs>
using Hessian = std::conditional_t<handy::IsSpecialization<::nlpp::impl::FirstArg<Fs...>, impl::Hessian>::value, ::nlpp::impl::FirstArg<Fs...>, impl::Hessian<impl::Visitor<Fs...>>>;

/** @name 
 *  @brief Functions used only to delegate the call with automatic type deduction
*/
//@{
template <class F>
Function<F> function (const F& f)
{
    return Function<F>(f);
}

template <class... Fs>
Gradient<Fs...> gradient (const Fs&... fs)
{
    return Gradient<Fs...>(fs...);
}

template <class... Fs>
FunctionGradient<Fs...> functionGradient (const Fs&... fs)
{
    return FunctionGradient<Fs...>(fs...);
}

template <class... Fs>
Hessian<Fs...> hessian (const Fs&... fs)
{
    return hessian<Fs...>(fs...);
}


template <class V>
struct Builder
{
    template <class... Fs>
    static auto function (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using TFs = typename Impl::TFs;

        if constexpr(impl::HasOp<impl::IsFunction, V, TFs>)
            return ::nlpp::wrap::function(std::get<impl::OpId<impl::IsFunction, V, TFs>>(std::forward_as_tuple(fs...)));
        
        else if constexpr(impl::HasOp<impl::IsFuncGrad, V, TFs>)
            return ::nlpp::wrap::function(std::get<impl::OpId<impl::IsFuncGrad, V, TFs>>(std::forward_as_tuple(fs...)));

        else
            static_assert(::nlpp::impl::always_false<::nlpp::impl::NthArg<0, Fs...>>, "The functor has no interface for the given parameter");
    }

    template <class... Fs>
    static auto gradient (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using TFs = typename Impl::TFs;

        if constexpr(impl::HasOp<impl::IsGradient, V, TFs> || impl::HasOp<impl::IsFuncGrad, V, TFs>)
            return ::nlpp::wrap::gradient(fs...);

        else if(impl::HasOp<impl::IsFunction, V, TFs>)
            return ::nlpp::wrap::gradient(::nlpp::fd::Gradient<Impl, ::nlpp::fd::Forward, ::nlpp::fd::AutoStep>(Impl(fs...)));

        else
            static_assert(::nlpp::impl::always_false<::nlpp::impl::NthArg<0, Fs...>>, "The functor has no interface for the given parameter");
    }

    template <class... Fs>
    static auto functionGradient (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using TFs = typename Impl::TFs;

        if constexpr(impl::HasOp<impl::IsFuncGrad, V, TFs> || (impl::HasOp<impl::IsFunction, V, TFs> && impl::HasOp<impl::IsGradient, V, TFs>))
            return ::nlpp::wrap::functionGradient(fs...);

        else if constexpr(impl::HasOp<impl::IsFunction, V, TFs>)
            return ::nlpp::wrap::functionGradient(fs..., ::nlpp::fd::Gradient<Impl, ::nlpp::fd::Forward, ::nlpp::fd::AutoStep>(Impl(fs...)));

        else
            static_assert(::nlpp::impl::always_false<::nlpp::impl::NthArg<0, Fs...>>, "The functor has no interface for the given parameter");
    }

    template <class... Fs>
    static auto hessian (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using TFs = typename Impl::TFs;

        if constexpr(impl::HasOp<impl::IsHessian, V, TFs>)
            return ::nlpp::wrap::hessian(fs...);

        // else if constexpr(HasOp::Gradient)
        //     return ::nlpp::wrap::hessian(::nlpp::fd::Hessian<Impl, ::nlpp::fd::Forward, ::nlpp::fd::AutoStep, double>(Impl(fs...)));

        else if constexpr(impl::HasOp<impl::IsFunction, V, TFs>)
            return ::nlpp::wrap::hessian(::nlpp::fd::Hessian<Impl, ::nlpp::fd::Forward, ::nlpp::fd::AutoStep, double>(Impl(fs...)));

        else
            static_assert(::nlpp::impl::always_false<::nlpp::impl::NthArg<0, Fs...>>, "The functor has no interface for the given parameter");
    }
};

template <class V, class... Fs>
auto makeFunc (const Fs&... fs)
{
    return Builder<V>::function(fs...);
}

template <class V, class... Fs>
auto makeGrad (const Fs&... fs)
{
    return Builder<V>::gradient(fs...);
}

template <class V, class... Fs>
auto makeFuncGrad (const Fs&... fs)
{
    return Builder<V>::functionGradient(fs...);
}

template <class V, class... Fs>
auto makeHessian (const Fs&... fs)
{
    return Builder<V>::hessian(fs...);
}


namespace poly
{

using ::nlpp::impl::Scalar, ::nlpp::impl::Plain, ::nlpp::impl::Plain2D;

template <class V>
using FunctionBase = std::function<Scalar<V>(const Plain<V>&)>;

template <class V>
using GradientBase = std::function<void(const Plain<V>&, Plain<V>&)>;

template <class V>
using FunctionGradientBase = std::function<Scalar<V>(const Plain<V>&, Plain<V>&, bool)>;

template <class V>
using HessianBase = std::function<void(const Plain<V>&, Plain2D<V>&)>;

template <class V>
using Function = ::nlpp::wrap::Function<FunctionBase<V>>;

template <class V>
using Gradient = ::nlpp::wrap::Gradient<GradientBase<V>>;

template <class V>
using FunctionGradient = ::nlpp::wrap::FunctionGradient<FunctionGradientBase<V>>;

template <class V>
using Hessian = ::nlpp::wrap::Hessian<HessianBase<V>>;


template <class V>
struct Builder
{
    template <class... Fs>
    static Function<V> function (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using TFs = typename Impl::TFs;

        if constexpr(impl::HasOp<impl::IsFunction, V, TFs>)
            return ::nlpp::wrap::function(FunctionBase<V>(std::get<impl::OpId<impl::IsFunction, V, TFs>>(std::forward_as_tuple(fs...))));

        else
            return ::nlpp::wrap::function(FunctionBase<V>(
                [func = ::nlpp::wrap::Builder<V>::function(fs...)]
                (const Plain<V>& x) -> Scalar<V> {
                    return func(x);
                }));
    }

    template <class... Fs>
    static Gradient<V> gradient (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using TFs = typename Impl::TFs;

        if constexpr(impl::HasOp<impl::IsGradient_0, V, TFs>)
            return ::nlpp::wrap::gradient(GradientBase<V>(std::get<impl::OpId<impl::IsGradient_0, V, TFs>>(std::forward_as_tuple(fs...))));

        else
            return ::nlpp::wrap::gradient(GradientBase<V>(
                [grad = ::nlpp::wrap::Builder<V>::gradient(fs...)]
                (const Plain<V>& x, Plain<V>& g) {
                    grad(x, g);
                }));
    }

    template <class... Fs>
    static FunctionGradient<V> functionGradient (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using TFs = typename Impl::TFs;

        if constexpr(impl::HasOp<impl::IsFuncGrad_0, V, TFs>)
            return ::nlpp::wrap::functionGradient(FunctionGradientBase<V>(std::get<impl::OpId<impl::IsFuncGrad_0, V, TFs>>(std::forward_as_tuple(fs...))));

        else
            return ::nlpp::wrap::functionGradient(FunctionGradientBase<V>(
                [funcGrad = ::nlpp::wrap::Builder<V>::functionGradient(fs...)]
                (const Plain<V>& x, Plain<V>& g, bool calcGrad) -> Scalar<V> {
                    return funcGrad(x, g, calcGrad);
                }));
    }

    template <class... Fs>
    static Hessian<V> hessian (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using TFs = typename Impl::TFs;

        if constexpr(impl::HasOp<impl::IsHessian_0, V, TFs>)
            return ::nlpp::wrap::hessian(HessianBase<V>(std::get<impl::OpId<impl::IsHessian_0, V, TFs>>(std::forward_as_tuple(fs...))));

        else
            return ::nlpp::wrap::hessian(HessianBase<V>(
                [hess = ::nlpp::wrap::Builder<V>::hessian(fs...)]
                (const Plain<V>& x, Plain2D<V>& h) {
                    hess(x, h);
                }));
    }
};

} // namespace poly

} // namespace nlpp::wrap
