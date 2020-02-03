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
                Gradient_1 = GetOpId<IsGradient_1, V, TFs>,  Gradient_2 = GetOpId<IsGradient_2, V, TFs>,
                Directional = GetOpId<IsDirectional, V, TFs>,
            FuncGrad = GetOpId<IsFuncGrad, V, TFs>, FuncGrad_0 = GetOpId<IsFuncGrad_0, V, TFs>,
                FuncGrad_1 = GetOpId<IsFuncGrad_1, V, TFs>,  FuncGrad_2 = GetOpId<IsFuncGrad_2, V, TFs>,
            Hessian = GetOpId<IsHessian, V, TFs>, Hessian_0 = GetOpId<IsHessian_0, V, TFs>,
                Hessian_1 = GetOpId<IsHessian_1, V, TFs>,  Hessian_2 = GetOpId<IsHessian_2, V, TFs>
        };
    };

    template <class V>
    struct HasOp
    {
        enum : bool
        {
            Function = OpId<V>::Function >= 0,
            Gradient = OpId<V>::Gradient >= 0, Gradient_0 = OpId<V>::Gradient_0 >= 0,
                Gradient_1 = OpId<V>::Gradient_1 >= 0, Gradient_2 = OpId<V>::Gradient_2 >= 0,
                Directional = OpId<V>::Directional >= 0,
            FuncGrad = OpId<V>::FuncGrad >= 0, FuncGrad_0 = OpId<V>::FuncGrad_0 >= 0,
                FuncGrad_1 = OpId<V>::FuncGrad_1 >= 0, FuncGrad_2 = OpId<V>::FuncGrad_2 >= 0, 
            Hessian = OpId<V>::Hessian >= 0, Hessian_0 = OpId<V>::Hessian_0 >= 0,
                Hessian_1 = OpId<V>::Hessian_1 >= 0, Hessian_2 = OpId<V>::Hessian_2 >= 0
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


    template <class V, bool Enable = HasOp<V>::Hessian_0, std::enable_if_t<Enable, int> = 0>
    void hessian (const Eigen::MatrixBase<V>& x, Plain2D<V>& h) const
    {
        hessianCall(std::get<OpId<V>::Hessian_0>(fs), x, h);
    }

    template <class V, bool Enable = HasOp<V>::Hessian_1, std::enable_if_t<Enable, int> = 0>
    Plain2D<V> hessian (const Eigen::MatrixBase<V>& x) const
    {
        return hessianCall(std::get<OpId<V>::Hessian_1>(fs), x);
    }

    template <class V, bool Enable = HasOp<V>::Hessian_2, std::enable_if_t<Enable, int> = 0>
    Plain<V> hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e) const
    {
        return hessianCall(std::get<OpId<V>::Hessian_2>(fs), x, e);
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


template <class Impl>
struct Hessian : public Impl
{
    template <class V>
    using HasOp = typename Impl:: template HasOp<V>;

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

    template <class... Fs>
    static auto hessian (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using HasOp = typename Impl:: template HasOp<V>;
        using OpId = typename Impl::template OpId<V>;

        if constexpr(HasOp::Hessian)
            return ::nlpp::wrap::hessian(fs...);

        // else if constexpr(HasOp::Gradient)
        //     return ::nlpp::wrap::hessian(::nlpp::fd::Hessian<Impl, ::nlpp::fd::Forward, ::nlpp::fd::AutoStep, double>(Impl(fs...)));

        else if constexpr(HasOp::Function)
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
        using HasOp = typename Impl:: template HasOp<V>;
        using OpId = typename Impl::template OpId<V>;

        if constexpr(HasOp::Function)
            return ::nlpp::wrap::function(FunctionBase<V>(std::get<OpId::Function>(std::forward_as_tuple(fs...))));

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
        using HasOp = typename Impl:: template HasOp<V>;
        using OpId = typename Impl::template OpId<V>;

        if constexpr(HasOp::Gradient_0)
            return ::nlpp::wrap::gradient(GradientBase<V>(std::get<OpId::Gradient_0>(std::forward_as_tuple(fs...))));

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

    template <class... Fs>
    static Hessian<V> hessian (const Fs&... fs)
    {
        using Impl = ::nlpp::wrap::impl::Visitor<Fs...>;
        using HasOp = typename Impl:: template HasOp<V>;
        using OpId = typename Impl::template OpId<V>;

        if constexpr(HasOp::Hessian_0)
            return ::nlpp::wrap::hessian(HessianBase<V>(std::get<OpId::Hessian_0>(std::forward_as_tuple(fs...))));

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
