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
      ::nlpp::impl::is_detected_v, ::nlpp::impl::always_false;

template <class T, class... Args>
using OperatorType = detected_t<std::invoke_result_t, T, Args...>;


template <class Impl, class V>
static constexpr bool isFunction = std::is_floating_point_v<OperatorType<Impl, V>>;

template <class Impl, class V>
static constexpr bool isGradient_0 = std::is_same_v<OperatorType<Impl, V, V&>, void>;

template <class Impl, class V>
static constexpr bool isGradient_1 = isMat<OperatorType<Impl, V>>;

template <class Impl, class V>
static constexpr bool isGradient_2 = std::is_same_v<OperatorType<Impl, V, V&, Scalar<V>>, void>;

template <class Impl, class V>
static constexpr bool isGradient = isGradient_0<Impl, V> || isGradient_1<Impl, V> || isGradient_2<Impl, V>;

template <class Impl, class V>
static constexpr bool isDirectional = std::is_floating_point_v<OperatorType<Impl, V, const V&>>;

template <class Impl, class V>
static constexpr bool isFuncGrad_0 = std::is_floating_point_v<OperatorType<Impl, V, V&, bool>>;

template <class Impl, class V>
static constexpr bool isFuncGrad_1 = std::is_floating_point_v<OperatorType<Impl, V, V&>>;

template <class Impl, class V>
static constexpr bool isFuncGrad_2 = std::is_floating_point_v<::nlpp::impl::NthArg<0, OperatorType<Impl, V>>> &&
                                     isMat<::nlpp::impl::NthArg<1, OperatorType<Impl, V>>>;

template <class Impl, class V>
static constexpr bool isFuncGrad = isFuncGrad_0<Impl, V> || isFuncGrad_1<Impl, V> || isFuncGrad_2<Impl, V>;

template <class Impl, class V, class U = Plain1D<V>>
static constexpr bool isHessian_1 = isMat<OperatorType<Impl, V, U>>;

template <class Impl, class V>
static constexpr bool isHessian_2 = isMat<OperatorType<Impl, V>> && OperatorType<Impl, V>::ColsAtCompileTime != 1;


template <class... Fs>
struct VisitorImpl : public Fs...
{
    VisitorImpl() {}
    VisitorImpl(const Fs... fs) : Fs(fs)...
    {
    }

    using Fs::operator()...;
};

template <class... Fs>
using Visitor = std::conditional_t<sizeof...(Fs) == 1, std::tuple_element_t<0, std::tuple<Fs...>>, VisitorImpl<Fs...>>;
 
/** @brief Function wrapping for user uniform defined function calculation
 * 
*/
template <class Impl_>
struct Function : public Impl_
{
    using Impl = Impl_;
    // using Impl::operator();

    Function (const Impl& impl) : Impl(impl) {}

    template <class V>
    Scalar<V> function (const Eigen::MatrixBase<V>& x);

    template <class V>
    Scalar<V> operator () (const Eigen::MatrixBase<V>& x)
    {
        return function(x);
    }

    /// Necessary to hide a lambda operator matching the exact arguments
    // template <typename T, int R, int C>
    // Scalar<V> operator () (const Eigen::Matrix<T, R, C>& x);
};


template <class... Impl_>
struct Gradient : public Visitor<Impl_...>
{
    using Impl = Visitor<Impl_...>;
    // using Impl::operator();
    // using Impl::Impl;

    // Gradient () {}
    Gradient (const Impl_&... impl) : Impl(impl...) {}

    // Gradient (const Impl& impl);

    template <class V>
    void gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g);

    template <class V>
    void gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V> fx);

    template <class V>
    Plain<V> gradient (const Eigen::MatrixBase<V>& x);

    template <class V>
    Scalar<V> directional (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e);

    template <class V>
    void operator() (const Eigen::MatrixBase<V>& x, Plain<V>& g)
    {
        return gradient(x, g);
    }

    template <class V>
    Plain<V> operator() (const Eigen::MatrixBase<V>& x)
    {
        return gradient(x);
    }

    template <class V>
    void operator() (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V> fx)
    {
        return gradient(x, g, fx);
    }
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
template <class... Impl_>
struct FunctionGradient : public Visitor<Impl_...>, public Function<Visitor<Impl_...>>, public Gradient<Impl_...>
{
    using Impl = Visitor<Impl_...>;
    // using Impl::Impl;
    using Impl::operator();
    using Func = Function<Visitor<Impl_...>>;
    using Grad = Gradient<Impl_...>;
    using Func::operator(), Func::function;
    using Grad::operator(), Grad::gradient;

    // FunctionGradient () {}
    FunctionGradient (const Impl_&... impl) : Impl(impl...), Func(Impl(impl...)), Grad(impl...)  {}



    // FunctionGradient (const Impl& impl);

    template <class V>
    Scalar<V> functionGradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad = true);

    template <class V>
    std::pair<Scalar<V>, Plain<V>> functionGradient (const Eigen::MatrixBase<V>& x);

    // template <class V>
    // Scalar<V> function (const Eigen::MatrixBase<V>& x);

    // template <class V>
    // void gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g);

    // template <class V>
    // Plain<V> gradient (const Eigen::MatrixBase<V>& x);

    template <class V>
    Scalar<V> operator() (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad = true)
    {
        return functionGradient(x, g, calcGrad);
    }

    template <class V>
    std::pair<Scalar<V>, Plain<V>> operator() (const Eigen::MatrixBase<V>& x)
    {
        return functionGradient(x);
    }
};

template <class Impl_>
struct Hessian : public Impl_
{
    using Impl = Impl_;

    Hessian (const Impl& impl);

    template <class V, class U>
    Plain<V> hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e);

    template <class V>
    Plain2D<V> hessian (const Eigen::MatrixBase<V>& x);

    template <class V, class U>
    Plain<V> operator() (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e);

    template <class V>
    Plain2D<V> operator() (const Eigen::MatrixBase<V>& x);
};

template <class Impl, class V>
Scalar<V> getFuncGrad (Impl& impl, const Eigen::MatrixBase<V>& x);

template <class Impl, class V>
Scalar<V> getFuncGrad (Impl& impl, const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad = true);

} // namespace impl


template <class Impl>
using Function = std::conditional_t<handy::IsSpecialization<Impl, impl::Function>::value, Impl, impl::Function<Impl>>;

template <class... Impl>
using Gradient = std::conditional_t<sizeof...(Impl) == 1 && handy::IsSpecialization<::nlpp::impl::FirstArg<Impl...>, impl::Gradient>::value, ::nlpp::impl::FirstArg<Impl...>, impl::Gradient<Impl...>>;

template <class... Impl>
using FunctionGradient = std::conditional_t<sizeof...(Impl) == 1 && handy::IsSpecialization<::nlpp::impl::FirstArg<Impl...>, impl::FunctionGradient>::value, ::nlpp::impl::FirstArg<Impl...>, impl::FunctionGradient<Impl...>>;

template <class Impl>
using Hessian = std::conditional_t<handy::IsSpecialization<Impl, impl::Hessian>::value, Impl, impl::Hessian<Impl>>;

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
template <class Impl>
Function<Impl> function (const Impl& impl)
{
    return Function<Impl>(impl);
}

/// Delegate the call to Gradient<Impl, Float>(impl)
template <class... Impl>
Gradient<Impl...> gradient (const Impl&... impl)
{
    return Gradient<Impl...>(impl...);
}

/** @brief Delegate the call to FunctionGradient where we have both the function and gradients in a single functor
 * 
 *  @param impl a functor having either <tt>Float operator()(const Vec&, Vec&)</tt> or
 *         <tt>std::pair<Float, Vec> operator()(const Vec&)</tt>
*/
template <class... Impl>
FunctionGradient<Impl...> functionGradient (const Impl&... impl)
{
    return FunctionGradient<Impl...>(impl...);
}

template <class Impl>
Hessian<Impl> hessian (const Impl& impl)
{
    return Hessian<Impl>(impl);
}


template <class V, class... Fs>
auto makeFuncGrad (const Fs&... fs)
{
    using Func = ::nlpp::wrap::impl::Visitor<Fs...>;

    if constexpr(::nlpp::wrap::impl::isFuncGrad<Func, V> || (::nlpp::wrap::impl::isFunction<Func, V> && ::nlpp::wrap::impl::isGradient<Func, V>))
        return functionGradient(fs...);
    
    else
        return functionGradient(fs..., ::nlpp::fd::Gradient<Func, ::nlpp::fd::Forward, ::nlpp::fd::AutoStep>(Func(fs...)));
}


namespace poly
{

using ::nlpp::impl::Scalar, ::nlpp::impl::Plain;

template <class V>
using Function = std::function<Scalar<V>(const V&)>;

template <class V>
using Gradient = std::function<void(const V&, V&)>;

template <class V>
using FunctionGradient = std::function<Scalar<V>(const V&, V&)>;


template <class V, class... Fs>
::nlpp::wrap::FunctionGradient<FunctionGradient<V>> makeFuncGrad (const Fs&... fs)
{
    using Func = ::nlpp::wrap::impl::Visitor<Fs...>;

    if constexpr(::nlpp::wrap::impl::isFuncGrad_0<Func, V>)
        return ::nlpp::wrap::functionGradient(FunctionGradient<V>(Func(fs...)));

    else if constexpr(::nlpp::wrap::impl::isFuncGrad<Func, V> || (::nlpp::wrap::impl::isFunction<Func, V> && ::nlpp::wrap::impl::isGradient<Func, V>))
        return ::nlpp::wrap::functionGradient(FunctionGradient<V>(
            [funcGrad = ::nlpp::wrap::functionGradient(fs...)]
            (const Plain<V>& x, Plain<V>& g, bool calcGrad) mutable -> Scalar<V> {
                return funcGrad(x, g, calcGrad);
            }));

    else
        return ::nlpp::wrap::functionGradient(FunctionGradient<V>(
            [funcGrad = ::nlpp::wrap::functionGradient(fs..., ::nlpp::fd::Gradient<Func, ::nlpp::fd::Forward, ::nlpp::fd::AutoStep>(Func(fs...)))]
            (const Plain<V>& x, Plain<V>& g, bool calcGrad) mutable -> Scalar<V> {
                return funcGrad(x, g, calcGrad);
            }));
}

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
