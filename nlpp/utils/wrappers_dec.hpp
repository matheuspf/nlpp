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

using ::nlpp::impl::Scalar, ::nlpp::impl::Plain, ::nlpp::impl::Plain1D,
      ::nlpp::impl::Plain2D, ::nlpp::impl::isMat, ::nlpp::impl::detected_t,
      ::nlpp::impl::is_detected_v, ::nlpp::impl::always_false, ::nlpp::impl::NthArg;


template <template <class, class> class Check, class V, class TFs, class Idx>
struct GetOpIdImpl;

template <template <class, class> class Check, class V, class... Fs, std::size_t... Is>
struct GetOpIdImpl<Check, V, std::tuple<Fs...>, std::index_sequence<Is...>>
{
    enum { value = (int(Check<Fs, V>::value * int(Is + 1)) + ...) - 1 };
};

template <template <class, class> class Check, class V, class TFs>
static constexpr int GetOpId = GetOpIdImpl<Check, V, TFs, std::make_index_sequence<std::tuple_size_v<TFs>>>::value;


template <class T, class... Args>
using OperatorType = detected_t<std::invoke_result_t, T, Args...>;


NLPP_MAKE_CALLER(function);
NLPP_MAKE_CALLER(gradient);
NLPP_MAKE_CALLER(funcGrad);


template <class Impl, class V>
struct IsFunction : std::bool_constant< std::is_floating_point_v<functionType<Impl, Plain<V>>> > {};



template <class Impl, class V>
struct IsGradient_0 : std::bool_constant< std::is_same_v<gradientType<Impl, Plain<V>, Plain<V>&>, void> > {};

template <class Impl, class V>
struct IsGradient_1 : std::bool_constant< isMat<gradientType<Impl, Plain<V>>> > {};

template <class Impl, class V>
struct IsGradient_2 : std::bool_constant< std::is_same_v<gradientType<Impl, Plain<V>, Plain<V>&, Scalar<V>>, void> > {};

template <class Impl, class V>
struct IsGradient : std::bool_constant< IsGradient_0<Impl, V>::value || IsGradient_1<Impl, V>::value || IsGradient_2<Impl, V>::value > {};

template <class Impl, class V>
struct IsDirectional : std::bool_constant< std::is_floating_point_v<gradientType<Impl, Plain<V>, Plain<V>>> > {};


template <class Impl, class V>
struct IsFuncGrad_0 : std::bool_constant< std::is_floating_point_v<funcGradType<Impl, Plain<V>, Plain<V>&, bool>> > {};

template <class Impl, class V>
struct IsFuncGrad_1 : std::bool_constant< std::is_floating_point_v<funcGradType<Impl, Plain<V>, Plain<V>&>> > {};

template <class Impl, class V>
struct IsFuncGrad_2 : std::bool_constant< std::is_floating_point_v<NthArg<0, funcGradType<Impl, Plain<V>>>> &&
                                          isMat<NthArg<1, funcGradType<Impl, Plain<V>>>> > {};

template <class Impl, class V>
struct IsFuncGrad : std::bool_constant< IsFuncGrad_0<Impl, V>::value || IsFuncGrad_1<Impl, V>::value || IsFuncGrad_2<Impl, V>::value > {};


template <class... Fs>
struct Visitor
{
    using TFs = std::tuple<Fs...>;

    // Visitor() {}

    Visitor(const Fs&... fs) : fs(fs...)
    {
    }

    template <class V>
    struct OpId
    {
        enum : int
        {
            Function = GetOpId<IsFunction, V, TFs>,
            Gradient = GetOpId<IsGradient, V, TFs>, Gradient_0 = GetOpId<IsGradient_0, V, TFs>, 
                Gradient_1 = GetOpId<IsGradient_1, V, TFs>,  Gradient_2 = GetOpId<IsGradient_2, V, TFs>, Directional = GetOpId<IsDirectional, V, TFs>,
            FuncGrad = GetOpId<IsFuncGrad, V, TFs>, FuncGrad_0 = GetOpId<IsFuncGrad_0, V, TFs>,
                FuncGrad_1 = GetOpId<IsFuncGrad_1, V, TFs>,  FuncGrad_2 = GetOpId<IsFuncGrad_2, V, TFs>
        };
    };

    template <class V>
    struct HasOp
    {
        enum : bool
        {
            Function = OpId<V>::Function >= 0,
            Gradient = OpId<V>::Gradient >= 0, Gradient_0 = OpId<V>::Gradient_0 >= 0,
                Gradient_1 = OpId<V>::Gradient_1 >= 0, Gradient_2 = OpId<V>::Gradient_2 >= 0, Directional = OpId<V>::Directional >= 0,
            FuncGrad = OpId<V>::FuncGrad >= 0, FuncGrad_0 = OpId<V>::FuncGrad_0 >= 0,
                FuncGrad_1 = OpId<V>::FuncGrad_1 >= 0, FuncGrad_2 = OpId<V>::FuncGrad_2 >= 0, 
        };
    };


    template <class V, bool Enable = HasOp<V>::Function, std::enable_if_t<Enable, int> = 0>
    Scalar<V> function (const Eigen::MatrixBase<V>& x) const
    {
        return functionCall(std::get<OpId<V>::Function>(fs), x);
    }

    template <class V, bool Enable = HasOp<V>::Gradient_0, std::enable_if_t<Enable, int> = 0>
    void gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g) const
    {
        gradientCall(std::get<OpId<V>::Gradient_0>(fs), x, g);
    }

    template <class V, bool Enable = HasOp<V>::Gradient_1, std::enable_if_t<Enable, int> = 0>
    Plain<V> gradient (const Eigen::MatrixBase<V>& x) const
    {
        return gradientCall(std::get<OpId<V>::Gradient_1>(fs), x);
    }

    template <class V, bool Enable = HasOp<V>::Gradient_2, std::enable_if_t<Enable, int> = 0>
    void gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V> fx) const
    {
        gradientCall(std::get<OpId<V>::Gradient_2>(fs), x, g, fx);
    }

    template <class V, bool Enable = HasOp<V>::Directional, std::enable_if_t<Enable, int> = 0>
    Scalar<V> gradient (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e) const
    {
        return gradientCall(std::get<OpId<V>::Directional>(fs), x, e);
    }


    template <class V, bool Enable = HasOp<V>::FuncGrad_0 || HasOp<V>::FuncGrad_1, std::enable_if_t<Enable, int> = 0>
    Scalar<V> funcGrad (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad =true) const
    {
        if constexpr(HasOp<V>::FuncGrad_0)
            return funcGradCall(std::get<OpId<V>::FuncGrad_0>(fs), x, g, calcGrad);

        else
            return funcGradCall(std::get<OpId<V>::FuncGrad_1>(fs), x, g);
    }

    // template <class V, bool Enable = HasOp<V>::FuncGrad_1, std::enable_if_t<Enable, int> = 0>
    // Scalar<V> funcGrad (const Eigen::MatrixBase<V>& x, Plain<V>& g) const
    // {
    //     return funcGradCall(std::get<OpId<V>::FuncGrad_1>(fs), x, g);
    // }

    template <class V, bool Enable = HasOp<V>::FuncGrad_2, std::enable_if_t<Enable, int> = 0>
    std::pair<Scalar<V>, Plain<V>> funcGrad (const Eigen::MatrixBase<V>& x) const
    {
        return funcGradCall(std::get<OpId<V>::FuncGrad_2>(fs), x);
    }


    template <class V, bool Enable = HasOp<V>::FuncGrad, std::enable_if_t<Enable, int> = 0>
    Scalar<V> getFuncGrad (const Eigen::MatrixBase<V>& x) const
    {
        static Plain<V> g;

        if(!HasOp<V>::FuncGrad_0 && !HasOp<V>::FuncGrad_2)
            g.resize(x.rows());

        return getFuncGrad(x, g, false);
    }

    template <class V, bool Enable = HasOp<V>::FuncGrad, std::enable_if_t<Enable, int> = 0>
    Scalar<V> getFuncGrad (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad = true) const
    {
        if constexpr(HasOp<V>::FuncGrad_0)
            return funcGradCall(std::get<OpId<V>::FuncGrad_0>(fs), x, g, calcGrad);

        else if constexpr(HasOp<V>::FuncGrad_1)
            return funcGradCall(std::get<OpId<V>::FuncGrad_1>(fs), x, g);

        else if constexpr(HasOp<V>::FuncGrad_2)
        {
            Scalar<V> f;
            std::tie(f, g) = funcGradCall(std::get<OpId<V>::FuncGrad_2>(fs), x);
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
    template <class V>
    using HasOp = typename Impl:: template HasOp<V>;

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
    template <class V>
    using HasOp = typename Impl:: template HasOp<V>;

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


/** @name 
 *  @brief The uniform interface wrapper for function/gradient functors
 *  
 *  @details This class provides the uniform interface where, given an user defined function/gradient functor or
 *           both function and gradient functors separated, you can call for function/gradient, function only pr
 *           gradient only, always avoiding to execute additional function calls when possible.
*/
//@{

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
    template <class V>
    using HasOp = typename Impl:: template HasOp<V>;

    template <typename... Args>
    FunctionGradient (Args&&... args) : impl(std::forward<Args>(args)...), func(std::forward<Args>(args)...), grad(std::forward<Args>(args)...) {}


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

    Function<Impl> func;
    Gradient<Impl> grad;

    //Function<std::reference_wrapper<const Impl>> func;
    //Gradient<std::reference_wrapper<const Impl>> grad;
};


//template <class Impl_>
//struct Hessian : public Impl_
//{
//    using Impl = Impl_;
//
//    Hessian (const Impl& impl);
//
//    template <class V, class U>
//    Plain<V> hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e);
//
//    template <class V>
//    Plain2D<V> hessian (const Eigen::MatrixBase<V>& x);
//
//    template <class V, class U>
//    Plain<V> operator() (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e);
//
//    template <class V>
//    Plain2D<V> operator() (const Eigen::MatrixBase<V>& x);
//};


} // namespace impl


template <class F>
using Function = std::conditional_t<handy::IsSpecialization<F, impl::Function>::value, F, impl::Function<impl::Visitor<F>>>;

template <class... Fs>
using Gradient = std::conditional_t<handy::IsSpecialization<::nlpp::impl::FirstArg<Fs...>, impl::Gradient>::value, ::nlpp::impl::FirstArg<Fs...>, impl::Gradient<impl::Visitor<Fs...>>>;

template <class... Fs>
using FunctionGradient = std::conditional_t<handy::IsSpecialization<::nlpp::impl::FirstArg<Fs...>, impl::FunctionGradient>::value, ::nlpp::impl::FirstArg<Fs...>, impl::FunctionGradient<impl::Visitor<Fs...>>>;


// template <class Impl>
// using Hessian = std::conditional_t<handy::IsSpecialization<Impl, impl::Hessian>::value, Impl, impl::Hessian<Impl>>;

/** @brief Alias for impl::FunctionGradient
 *  @details There are four conditions:
 *           - Is @c Grad given?
 *              - If not, is @c Func already an impl::FunctionGradient?
 *                  - (1) If so, simply set the result to itself (to avoid multiple wrapping)
 *                  - Otherwise, is @c Func a function but not a function/gradient functor?
 *                      - (2) If so, set the result to impl::FunctionGradient<Func, fd::Gradient<Func>> (use finite difference to aproximate the gradient)
 *                      - (3) Otherwise it should be a function/gradient functor, so we use impl::FunctionGradient<Func>
 *              - (4) If yes, set the result to impl::FunctionGradient<Func, Grad>
*/
// template <class Func, class Grad = void>
// using FunctionGradient = std::conditional_t<std::is_same<Grad, void>::value,
//     std::conditional_t<handy::IsSpecialization<Func, impl::FunctionGradient>::value,
//         Func,
//         // std::conditional_t<wrap::FunctionType<Func>::value >= 0 && wrap::FunctionGradientType<Func>::value < 0,
//         //     impl::FunctionGradient<Func, fd::Gradient<Func>>,
//             impl::FunctionGradient<Func>
//         // >
//     >,
//     impl::FunctionGradient<Func, Grad>
// >;


// template <class Impl, class... Impls>
// using FunctionGradient = std::conditional_t<sizeof...(Impls) == 0 && handy::IsSpecialization<Impl, impl::FunctionGradient>::value,
//     Impl,
//     std::conditional_t<!impl::isGradient<impl::Visitor<Impl, Impls...>> && !impl::isFuncGrad<impl::Visitor<Impl, Impls...>>,
//         impl::FunctionGradient<Impl, Impls..., ::nlpp::fd::Gradient<impl::Visitor<Impl, Impls...>, ::nlpp::fd::Forward, ::nlpp::fd::AutoStep>,
//         impl::FunctionGradient<Impl, Impls...>>>;

/** @name 
 *  @brief Functions used only to delegate the call with automatic type deduction
*/
//@{

/// Delegate the call to Function<Impl, Float>(impl)
template <class F>
Function<F> function (const F& f)
{
    return Function<F>(f);
}

/// Delegate the call to Gradient<Impl, Float>(impl)
template <class... Fs>
Gradient<Fs...> gradient (const Fs&... fs)
{
    return Gradient<Fs...>(fs...);
}

/** @brief Delegate the call to FunctionGradient where we have both the function and gradients in a single functor
 * 
 *  @param impl a functor having either <tt>Float operator()(const Vec&, Vec&)</tt> or
 *         <tt>std::pair<Float, Vec> operator()(const Vec&)</tt>
*/

template <class... Fs>
FunctionGradient<Fs...> functionGradient (const Fs&... fs)
{
    return FunctionGradient<Fs...>(fs...);
}


// template <class Impl>
// Hessian<Impl> hessian (const Impl& impl)
// {
//     return Hessian<Impl>(impl);
// }


template <class V>
struct Builder
{
    template <class... Fs>
    static auto function (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using HasOp = typename Impl:: template HasOp<V>;
        using OpId = typename Impl::template OpId<V>;

        if constexpr(HasOp::Function)
            return ::nlpp::wrap::function(std::get<OpId::Function>(std::forward_as_tuple(fs...)));
        
        else if constexpr(HasOp::FuncGrad)
            return ::nlpp::wrap::function(std::get<OpId::FuncGrad>(std::forward_as_tuple(fs...)));

        else
            static_assert(::nlpp::impl::always_false<::nlpp::impl::NthArg<0, Fs...>>, "The functor has no interface for the given parameter");
    }

    template <class... Fs>
    static auto gradient (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using HasOp = typename Impl:: template HasOp<V>;
        using OpId = typename Impl::template OpId<V>;

        if constexpr(HasOp::Gradient || HasOp::FuncGrad)
            return ::nlpp::wrap::gradient(fs...);

        else if(HasOp::Function)
            return ::nlpp::wrap::gradient(::nlpp::fd::Gradient<Impl, ::nlpp::fd::Forward, ::nlpp::fd::AutoStep>(Impl(fs...)));

        else
            static_assert(::nlpp::impl::always_false<::nlpp::impl::NthArg<0, Fs...>>, "The functor has no interface for the given parameter");
    }

    template <class... Fs>
    static auto functionGradient (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using HasOp = typename Impl:: template HasOp<V>;
        using OpId = typename Impl::template OpId<V>;

        if constexpr(HasOp::FuncGrad || (HasOp::Function && HasOp::Gradient))
            return ::nlpp::wrap::functionGradient(fs...);

        else if constexpr(HasOp::Function)
            return ::nlpp::wrap::functionGradient(fs..., ::nlpp::fd::Gradient<Impl, ::nlpp::fd::Forward, ::nlpp::fd::AutoStep>(Impl(fs...)));

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


namespace poly
{

using ::nlpp::impl::Scalar, ::nlpp::impl::Plain;

template <class V>
using FunctionBase = std::function<Scalar<V>(const Plain<V>&)>;

template <class V>
using GradientBase = std::function<void(const Plain<V>&, Plain<V>&)>;

template <class V>
using FunctionGradientBase = std::function<Scalar<V>(const Plain<V>&, Plain<V>&, bool)>;

template <class V>
using Function = ::nlpp::wrap::Function<FunctionBase<V>>;

template <class V>
using Gradient = ::nlpp::wrap::Gradient<GradientBase<V>>;

template <class V>
using FunctionGradient = ::nlpp::wrap::FunctionGradient<FunctionGradientBase<V>>;


template <class V>
struct Builder
{
    template <class... Fs>
    static Function<V> function (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using HasOp = typename Impl:: template HasOp<V>;
        using OpId = typename Impl::template OpId<V>;

        if constexpr(HasOp::Function)
            return ::nlpp::wrap::function(FunctionBase<V>(std::get<OpId::Function>(std::forward_as_tuple(fs...))));

        else
            return ::nlpp::wrap::function(FunctionBase<V>(
                [func = ::nlpp::wrap::Builder<V>::function(fs...)]
                (const Plain<V>& x) -> Scalar<V> {
                    return funcGrad(x);
                }));
    }

    template <class... Fs>
    static Gradient<V> gradient (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using HasOp = typename Impl:: template HasOp<V>;
        using OpId = typename Impl::template OpId<V>;

        if constexpr(HasOp::Gradient_0)
            return ::nlpp::wrap::gradient(GradientBase<V>(std::get<OpId::Gradient_0>(std::forward_as_tuple(fs...))));

        else
            return ::nlpp::wrap::gradient(GradientBase<V>(
                [func = ::nlpp::wrap::Builder<V>::gradient(fs...)]
                (const Plain<V>& x, Plain<V>& g) {
                    funcGrad(x, g);
                }));
    }

    template <class... Fs>
    static FunctionGradient<V> functionGradient (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using HasOp = typename Impl:: template HasOp<V>;
        using OpId = typename Impl::template OpId<V>;

        if constexpr(HasOp::FuncGrad_0)
            return ::nlpp::wrap::functionGradient(FunctionGradientBase<V>(std::get<OpId::FuncGrad_0>(std::forward_as_tuple(fs...))));

        else
            return ::nlpp::wrap::functionGradient(FunctionGradientBase<V>(
                [funcGrad = ::nlpp::wrap::Builder<V>::functionGradient(fs...)]
                (const Plain<V>& x, Plain<V>& g, bool calcGrad) -> Scalar<V> {
                    return funcGrad(x, g, calcGrad);
                }));
    }
};

} // namespace poly


//@}
//@}
/*
namespace poly
{

template <class V_ = ::nlpp::Vec>
struct Function
{
    using V = V_;
    using Float = Scalar<V>;

    using FuncType = Float (const V&);


    Function () {}

    Function(const std::function<FuncType>& func) : func(func)
    {
    }

    Float function (const V& x)
    {
        return func(x);
    }

    Float operator () (const V& x)
    {
        return function(x);
    }

    std::function<FuncType> func;
};


template <class V_ = Vec>
struct Gradient
{
    using V = V_;
    using Float = ::nlpp::impl::Scalar<V>;

    using GradType_0 = void (const V&, Plain<V>>&);
    using GradType_1 = V (const V&);
    using GradType_2 = void (const V&, Plain<V>>&, Float);
    using GradType_2 = void (const V&, Plain<V>>&, Float);

    Gradient () {}

    template <class G>
    Gradient (const G& grad) : grad(grad) {}


    void gradient (const V& x, Plain<V>& g)
    {
        switch(grad.index()):
        {
            case 0:
                std::get<GradType_0>(grad)(x, g);
                break;
            case 1:
                g = std::get<GradType_0>(grad)(x);
                break;
            case 2:
                std::get<GradType_0>(grad)(x, g, std::nan("0"));
                break;
            default:
                std::assert("Class was not initialized");
        }
    }

    V gradient (const V& x)
    {
        switch(grad.index()):
        {
            case 0:
                Plain<V> g(x.rows());
                std::get<GradType_0>(grad)(x, g);
                return g;
            case 1:
                return std::get<GradType_1>(grad)(x);
            case 2:
                Plain<V> g(x.rows());
                std::get<GradType_0>(grad)(x, g, std::nan("0"));
                return g;
            default:
                std::assert("Class was not initialized");
        }
    }

    void gradient (const V& x, Plain<V>& g, Scalar<V> fx)
    {
        switch(grad.index()):
        {
            case 0:
                std::get<GradType_0>(grad)(x, g);
                break;
            case 1:
                g = std::get<GradType_0>(grad)(x);
                break;
            case 2:
                std::get<GradType_0>(grad)(x, g, fx);
                break;
            default:
                std::assert("Class was not initialized");
        }
    }

    void operator () (const V& x, Plain<V>>& g)
    {
        gradient(x, g);
    }

    V operator () (const V& x)
    {
        return gradient(x);
    }

    void operator () (const V& x, Plain<V>>& g, Scalar<V> fx)
    {
        gradient(x, g, fx);
    }

    Float directional (const V& x, const V& e, Float fx)
    {
        return direc(x, e, fx);
    }


    void init ()
    {
        if(!grad_1)
        {
            grad_1 = [gradImpl = nlpp::wrap::gradient(grad_2)](const V& x) mutable -> V
            {
                return gradImpl(x);
            };
        }

        if(!grad_2)
        {
            grad_2 = [gradImpl = nlpp::wrap::gradient(grad_1)](const V& x, Plain<V>>& g) mutable
            {
                gradImpl(x, g);
            };
        }
    }

    std::variant<std::function<GradType_0>,
                 std::function<GradType_1>,
                 std::function<GradType_2>> grad;
};


template <class V_ = Vec>
struct FunctionGradient : public Function<V_>, public Gradient<V_>
{
    using V = V_;
    using Float = ::nlpp::impl::Scalar<V>;

    using Func = Function<V>;
    using Grad = Gradient<V>;

    using Func::function;
    using Func::func;
    using Grad::gradient;
    using Grad::grad_1;
    using Grad::grad_2;

    using FuncType = typename Func::FuncType;
    using GradType_1 = typename Grad::GradType_1;
    using GradType_2 = typename Grad::GradType_2;

    using FuncGradType_1 = std::pair<Float, V> (const V&);
    using FuncGradType_2 = Float (const V&, Plain<V>>&);
    
    using DirectionalType = Float (const V&, const V&, Float);


    FunctionGradient (const std::function<FuncGradType_1>& grad_1, ::nlpp::impl::Precedence<0>) : funcGrad_1(funcGrad_1)
    {
        init();
    }

    FunctionGradient (const std::function<FuncGradType_2>& grad_2, ::nlpp::impl::Precedence<1>) : funcGrad_2(funcGrad_2)
    {
        init();
    }

    FunctionGradient (const std::function<FuncType>& func, const std::function<GradType_1>& grad_1, ::nlpp::impl::Precedence<0>) : Func(func), Grad(grad_1)
    {
        init();
    }

    FunctionGradient (const std::function<FuncType>& func, const std::function<GradType_2>& grad_2, ::nlpp::impl::Precedence<1>) : Func(func), Grad(grad_2)
    {
        init();
    }

    FunctionGradient (const std::function<FuncType>& func, ::nlpp::impl::Precedence<0>) : Func(func)
    {
        init();
    }


    template <class F, std::enable_if_t<(::nlpp::wrap::FunctionType<F>::value >= 0) || (::nlpp::wrap::FunctionGradientType<F>::value >= 0), int> = 0>
    FunctionGradient(const F& func) : FunctionGradient(func, ::nlpp::impl::Precedence<0>{}) {}
    
    template <class F, class G, std::enable_if_t<(::nlpp::wrap::FunctionType<F>::value >= 0) && (::nlpp::wrap::GradientType<G>::value >= 0), int> = 0>
    FunctionGradient(const F& func, const G& grad) : FunctionGradient(func, grad, ::nlpp::impl::Precedence<0>{}) {}



    std::pair<Float, V> functionGradient (const V& x)
    {
        return funcGrad_1(x);
    }

    Float functionGradient (const V& x, Plain<V>>& g)
    {
        return funcGrad_2(x, g);
    }


    std::pair<Float, V> operator () (const V& x)
    {
        return functionGradient(x);
    }

    Float operator () (const V& x, Plain<V>>& g)
    {
        return functionGradient(x.eval(), g);
    }


    Float directional (const V& x, const V& e)
    {
        return directional(x, e, func(x));
    }

    Float directional (const V& x, const V& e, Float fx)
    {
        return direc(x, e, fx);
    }


    template <class FuncGradImpl>
    void setFuncGrad(FuncGradImpl funcGradImpl)
    {
        setFuncGrad_1(funcGradImpl);
        setFuncGrad_2(funcGradImpl);
    }

    template <class FuncGradImpl>
    void setFuncGrad_1(FuncGradImpl funcGradImpl)
    {
        funcGrad_1 = [funcGradImpl](const V& x) mutable -> std::pair<Float, V>
            {
                return funcGradImpl(x);
            };
    }
    
    template <class FuncGradImpl>
    void setFuncGrad_2(FuncGradImpl funcGradImpl)
    {
        funcGrad_2 = [funcGradImpl](const V& x, Plain<V>>& g) mutable -> Float
        {
            return funcGradImpl(x, g);
        }; 
    }


    void init ()
    {
        if(!funcGrad_1 && !funcGrad_2)
        {
            if(grad_1)
                setFuncGrad(::nlpp::wrap::functionGradient(func, grad_1));

            else if(grad_2)
                setFuncGrad(::nlpp::wrap::functionGradient(func, grad_2));

            else
                setFuncGrad(::nlpp::wrap::functionGradient(func));
        }

        else if(funcGrad_1 && !funcGrad_2)
            setFuncGrad_1(::nlpp::wrap::functionGradient(funcGrad_1));

        else if(!funcGrad_1 && funcGrad_2)
            setFuncGrad_2(::nlpp::wrap::functionGradient(funcGrad_2));

        if(!func)
        {
            if(funcGrad_1)
                func = [funcGrad = funcGrad_1](const V& x) -> Float
                {
                    return std::get<0>(funcGrad(x));
                };

            else if(funcGrad_2)
                func = [funcGrad = funcGrad_2](const V& x) -> Float
                {
                    V g(x.rows(), x.cols());
                    return funcGrad(x, g);
                };
        }

        if(!grad_1)
        {
            if(grad_2)
                grad_1 = [gradImpl = nlpp::wrap::gradient(grad_2)](const V& x) mutable -> V
                {
                    return gradImpl(x);
                };
            
            else
                grad_1 = [gradImpl = funcGrad_1](const V& x) mutable -> V
                {
                    return std::get<1>(gradImpl(x));
                };
        }
        
        if(!grad_2)
        {
            if(grad_1)
                grad_2 = [gradImpl = nlpp::wrap::gradient(grad_1)](const V& x, Plain<V>>& g) mutable
                {
                    gradImpl(x, g);
                };
            
            else
                grad_2 = [gradImpl = funcGrad_2](const V& x, Plain<V>>& g) mutable
                {
                    gradImpl(x, g);
                };
        }

        direc = [funcGrad = ::nlpp::wrap::functionGradient(func)](const V& x, const V& e, Float fx) mutable -> Float
        {
            return funcGrad.directional(x, e, fx);
        };
    }



    std::function<FuncGradType_1> funcGrad_1;
    std::function<FuncGradType_2> funcGrad_2;
    std::function<DirectionalType> direc;
};


template <class V_ = ::nlpp::Vec, class M_ = ::nlpp::Mat>
struct Hessian
{
    using V = V_;
    using M = M_;
    using Float = ::nlpp::impl::Scalar<V>;

    using HessType = M (const Eigen::MatrixBase<V>&);

    Hessian (const std::function<HessType>& hessian) : hessian(hessian) {}


    M operator () (const V& x)
    {
        return hessian(x);
    }


    std::function<HessType> hessian;
};

} // namespace poly
*/
//@}

} // namespace nlpp::wrap
