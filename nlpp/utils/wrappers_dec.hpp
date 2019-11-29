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
#include "utils/finiteDifference_dec.hpp"


namespace nlpp
{

/// Wrap namespace
namespace wrap
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
 
/// Check if class T is a function functor
template <class T, class V>
constexpr int functionType ();

/// Check if class T is a gradient functor
template <class T, class V>
constexpr int gradientType ();

/// Check if class T is a function/gradient functor
template <class T, class V>
constexpr int functionGradientType ();

/// Check if class T is a hessian functor
template <class T, class V>
constexpr int hessianType ();


template <class T, class V>
constexpr int functionType_v = functionType<T, V>();

template <class T, class V>
constexpr int gradientType_v = gradientType<T, V>();

template <class T, class V>
constexpr int functionGradientType_v = functionGradientType<T, V>();

template <class T, class V>
constexpr int hessianType_v = hessianType<T, V>();


template <class T, class V>
constexpr bool isFunction_v = functionType<T, V>();

template <class T, class V>
constexpr bool isGradient = gradientType<T, V>();

template <class T, class V>
constexpr bool isFunctionGradient = functionGradientType<T, V>();

template <class T, class V, class V2 = V>
constexpr bool isHessian = hessianType<T, V, V2>();



namespace impl
{

/** @brief Function wrapping for user uniform defined function calculation
 * 
*/
template <class Impl_>
struct Function : public Impl_
{
    using Impl = Impl_;
    
    Function (const Impl& impl);

    template <class V>
    auto function (const Eigen::MatrixBase<V>& x);

    template <class V>
    auto operator () (const Eigen::MatrixBase<V>& x);

    /// Necessary to hide a lambda operator matching the exact arguments
    template <typename T, int R, int C>
    auto operator () (const Eigen::Matrix<T, R, C>& x);
};

} // namespace impl

namespace impl
{

template <class Impl_>
struct Gradient : public Impl_
{
    using Impl = Impl_;
    
    Gradient (const Impl& impl) : Impl(impl) {}

    //static_assert(GradientType<Impl>::value >= 0, "The given functor does not have a gradient interface");


    template <class V, typename... Args>
    auto delegate (const Eigen::MatrixBase<V>& x, Args&&... args)
    {
        if constexpr(gradType == 0)
            return Impl::gradient(x, std::forward<decltype(args)>(args)...);
            
        else
            return Impl::operator()(x, std::forward<Args>(args)...);
    }

    template <class V, class I = Impl, std::enable_if_t<GradientType<I, V>::value < 2, int> = 0>
    void gradient (const Eigen::MatrixBase<V>& x, ::nlpp::impl::Plain<V>& g)
    {
        delegate(x, g);
    }

    template <class V, class I = Impl, std::enable_if_t<GradientType<I, V>::value < 2, int> = 0>
    ::nlpp::impl::Plain<V> gradient (const Eigen::MatrixBase<V>& x)
    {
        ::nlpp::impl::Plain<V> g(x.rows(), x.cols());

        delegate(x, g);

        return g;
    }


    template <class V, class I = Impl, std::enable_if_t<GradientType<I, V>::value >= 2, int> = 0>
    void gradient (const Eigen::MatrixBase<V>& x, ::nlpp::impl::Plain<V>& g)
    {
        g = delegate(x);
    }

    template <class V, class I = Impl, std::enable_if_t<GradientType<I, V>::value >= 2, int> = 0>
    ::nlpp::impl::Plain<V> gradient (const Eigen::MatrixBase<V>& x)
    {
        constexpr auto gradType = gradientType_v<Impl, V>;

        static_assert(isGradient_v<Impl, V>, "The functor has no interface for the given parameter")

        if constexpr (gradType < 2)
        {
            ::nlpp::impl::Plain<V> g(x.rows(), x.cols());
            return delegate(x, g);
        }

        else
            return delegate(x);

    }


    template <class V>
    void operator() (const Eigen::MatrixBase<V>& x, ::nlpp::impl::Plain<V>& g)
    {
        gradient(x, g);
    }

    template <class V>
    ::nlpp::impl::Plain<V> operator() (const Eigen::MatrixBase<V>& x)
    {
        return gradient(x);
    }
};

} // namespace impl




namespace impl
{

/** @name 
 *  @brief The uniform interface wrapper for function/gradient functors
 *  
 *  @details This class provides the uniform interface where, given an user defined function/gradient functor or
 *           both function and gradient functors separated, you can call for function/gradient, function only pr
 *           gradient only, always avoiding to execute additional function calls when possible.
*/
//@{

/// Forward declaration
template <class...>
struct FunctionGradient;


/** @brief Specialization for when both function and gradients are given separatelly
 * 
 *  @tparam Func A functor having <tt>Scalar Func::operator()(const Vec&)</tt> defined
 *  @tparam Func A functor having either <tt>Vec Func::operator()(const Vec&)</tt> or 
 *          <tt>void Func::operator()(const Vec&, Vec&)</tt> defined.
 * 
 *  @note This class inherits both the @c Func and @c Grad templates, wrapping the @c Grad 
 *        into @c Gradient<Grad> first
*/
template <class Func, class Grad>
struct FunctionGradient<Func, Grad> : public Function<Func>, public Gradient<Grad>
{
    using Function<Func>::function;
    using Gradient<Grad>::gradient;

    /// Single constructor, delegated to Func and Grad
    template <class F = Func, class G = Grad, std::enable_if_t<handy::HasConstructor<G>::value, int> = 0>
    FunctionGradient (const F& f = Func{}, const G& g = Grad{}) : Function<Func>(f), Gradient<Grad>(g) {}
    
    template <class F = Func, class G = Grad, std::enable_if_t<!handy::HasConstructor<G>::value, int> = 0>
    FunctionGradient (const F& f = Func{}, const G& g = Grad{Func{}}) : Function<Func>(f), Gradient<Grad>(g) {}


    /** @name
     *  @brief Operators for function/gradient calls.
     * 
     *  @details Returns both function and gradient if only a @Vec is given. If the reference @c g
     *           is also given, returns only the Func::operator()(x) return and calls 
     *           @c Gradient<Grad>::operator()(x, g) to write on @c g.
     * 
     *  @param x The vector for which we will be evaluating both function and gradient
     *  @param g The reference where we are going to store the gradient
    */
    //@{
    template <class V>
    auto functionGradient (const Eigen::MatrixBase<V>& x)
    {
        return std::make_pair(function(x), gradient(x));
    }

    template <class V>
    auto functionGradient (const Eigen::MatrixBase<V>& x, ::nlpp::impl::Plain<V>& g)
    {
        gradient(x, g);

        return function(x);
    }
    //@}



    template <class V>
    auto operator () (const Eigen::MatrixBase<V>& x)
    {
        return functionGradient(x);
    }

    /// Necessary to hide a lambda operator matching the exact arguments
    template <typename T, int R, int C>
    auto operator () (const Eigen::Matrix<T, R, C>& x)
    {
        return functionGradient(x);
    }

    template <class V>
    auto operator () (const Eigen::MatrixBase<V>& x, ::nlpp::impl::Plain<V>& g)
    {
        return functionGradient(x, g);
    }
};


template <class Func, template <class, class> class Difference, class Step>
struct FunctionGradient<Func, fd::Gradient<Func, Difference, Step>> : Function<Func>, public fd::Gradient<Func, Difference, Step>
{
    using Function = wrap::Function<Func>;
    using Gradient = fd::Gradient<Func, Difference, Step>;


    using Function::function;
    using Gradient::gradient;
    using Gradient::directional;


    FunctionGradient (const Function& f = Function{}) : Function(f), Gradient(f) {} 

    FunctionGradient (const Function& f, const Gradient& g) : Function(f), Gradient(Function(f)) {}
    

    /** @name
     *  @brief Operators for function/gradient calls.
     * 
     *  @details Returns both function and gradient if only a @Vec is given. If the reference @c g
     *           is also given, returns only the Func::operator()(x) return and calls 
     *           @c Gradient<Grad>::operator()(x, g) to write on @c g.
     * 
     *  @param x The vector for which we will be evaluating both function and gradient
     *  @param g The reference where we are going to store the gradient
    */
    //@{
    template <class V>
    auto functionGradient (const Eigen::MatrixBase<V>& x)
    {
        auto f = function(x);

        return std::make_pair(f, gradient(x, f));
    }

    template <class V>
    auto functionGradient (const Eigen::MatrixBase<V>& x, ::nlpp::impl::Plain<V>& g)
    {
        auto f = function(x);

        gradient(x, g, f);

        return f;
    }
    //@}

    template <class V>
    auto operator () (const Eigen::MatrixBase<V>& x)
    {
        return functionGradient(x);
    }

    /// Necessary to hide a lambda operator matching the exact arguments
    template <typename T, int R, int C>
    auto operator () (const Eigen::Matrix<T, R, C>& x)
    {
        return functionGradient(x);
    }

    template <class V>
    auto operator () (const Eigen::MatrixBase<V>& x, ::nlpp::impl::Plain<V>& g)
    {
        return functionGradient(x, g);
    }
};


template <class Impl_>
struct FunctionGradient<Impl_> : public Impl_
{
    using Impl = Impl_;
    
    FunctionGradient (const Impl& impl) : Impl(impl) {}

    //static_assert(FunctionGradientType<Impl>::value >= 0, "The given functor does not have a function/gradient interface");


    template <class V, typename... Args, class I = Impl, std::enable_if_t<FunctionGradientType<I, V>::value % 2 == 0, int> = 0>
    auto delegate (const Eigen::MatrixBase<V>& x, Args&&... args)
    {
        return Impl::functionGradient(x, std::forward<Args>(args)...);
    }

    template <class V, typename... Args, class I = Impl, std::enable_if_t<FunctionGradientType<I, V>::value % 2 != 0, int> = 0>
    auto delegate (const Eigen::MatrixBase<V>& x, Args&&... args)
    {
        return Impl::operator()(x, std::forward<Args>(args)...);
    }


    template <class V, class I = Impl, std::enable_if_t<FunctionGradientType<I, V>::value < 2, int> = 0>
    auto functionGradient (const Eigen::MatrixBase<V>& x, ::nlpp::impl::Plain<V>& g)
    {
        return delegate(x, g);
    }

    template <class V, class I = Impl, std::enable_if_t<FunctionGradientType<I, V>::value < 2, int> = 0>
    auto functionGradient (const Eigen::MatrixBase<V>& x)
    {
        ::nlpp::impl::Plain<V> g(x.rows(), x.cols());

        auto f = delegate(x, g);

        return std::make_pair(f, g);
    }


    template <class V, class I = Impl, std::enable_if_t<FunctionGradientType<I, V>::value >= 2, int> = 0>
    auto functionGradient (const Eigen::MatrixBase<V>& x, ::nlpp::impl::Plain<V>& g)
    {
        std::decay_t<decltype(std::get<0>(delegate(x)))> f;

        std::tie(f, g) = delegate(x);
        
        return f;
    }

    template <class V, class I = Impl, std::enable_if_t<FunctionGradientType<I, V>::value >= 2, int> = 0>
    auto functionGradient (const Eigen::MatrixBase<V>& x)
    {
        return delegate(x);
    }


    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x, ::nlpp::impl::Plain<V>& g)
    {
        return functionGradient(x, g);
    }

    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x)
    {
        return functionGradient(x);
    }

    /// Necessary to hide a lambda operator matching the exact arguments
    template <typename T, int R, int C>
    auto operator () (const Eigen::Matrix<T, R, C>& x)
    {
        return functionGradient(x);
    }


    template <class V, class I = Impl, std::enable_if_t<FunctionType<I, V>::value >= 0, int> = 0>
    auto function (const Eigen::MatrixBase<V>& x)
    {
        return Function<Impl>(*this)(x);
    }

    template <class V, class I = Impl, std::enable_if_t<FunctionType<I, V>::value < 0, int> = 0>
    auto function (const Eigen::MatrixBase<V>& x)
    {
        static typename V::PlainObject dummieG;

        return functionGradient(x, dummieG);
    }

    template <class V, class... Args, class I = Impl, std::enable_if_t<GradientType<I, V>::value >= 0, int> = 0>
    auto gradient (const Eigen::MatrixBase<V>& x, Args&&... args)
    {
        return Gradient<Impl>(*this)(x, std::forward<Args>(args)...);
    }

    template <class V, class... Args, class I = Impl, std::enable_if_t<GradientType<I, V>::value < 0, int> = 0>
    void gradient (const Eigen::MatrixBase<V>& x, ::nlpp::impl::Plain<V>& g)
    {
        functionGradient(x, g);
    }

    template <class V, class... Args, class I = Impl, std::enable_if_t<GradientType<I, V>::value < 0, int> = 0>
    auto gradient (const Eigen::MatrixBase<V>& x)
    {
        return std::get<1>(functionGradient(x));
    }
};

} // namespace impl


template <class Impl_>
struct Hessian : public Impl_
{
    using Impl = Impl_;
    
    Hessian (const Impl& impl) : Impl(impl) {}


    template <class V, typename... Args, class I = Impl, std::enable_if_t<HessianType<I, V>::value % 2 == 0, int> = 0>
    auto delegate (const Eigen::MatrixBase<V>& x, Args&&... args)
    {
        return Impl::hessian(x, std::forward<Args>(args)...);
    }

    template <class V, typename... Args, class I = Impl, std::enable_if_t<HessianType<I, V>::value % 2 == 1, int> = 0>
    auto delegate (const Eigen::MatrixBase<V>& x, Args&&... args)
    {
        return Impl::operator()(x, std::forward<Args>(args)...);
    }


    template <class V, class U, class I = Impl, std::enable_if_t<HessianType<I, V>::value < 2, int> = 0>
    ::nlpp::impl::Plain<V> hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e)
    {
        return delegate(x, e);
    }

    // template <class V, class I = Impl, std::enable_if_t<HessianType<I, V>::value < 2, int> = 0>
    // auto gradient (const Eigen::MatrixBase<V>& x)
    // {
    // }


    template <class V, class U, class I = Impl, std::enable_if_t<GradientType<I, V>::value >= 2, int> = 0>
    ::nlpp::impl::Plain<V> hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e)
    {
        return delegate(x) * e;
    }

    template <class V, class I = Impl, std::enable_if_t<GradientType<I, V>::value >= 2, int> = 0>
    ::nlpp::impl::Plain2D<V> hessian (const Eigen::MatrixBase<V>& x)
    {
        return delegate(x);
    }


    template <class V, class U>
    auto operator() (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e)
    {
        return hessian(x, e);
    }

    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x)
    {
        return hessian(x);
    }
};


/** @name 
 *  @brief Functions used only to delegate the call with automatic type deduction
*/
//@{

/// Delegate the call to Function<Impl, Float>(impl)
template <class Impl>
auto function (const Impl& impl)
{
    return Function<Impl>(impl);
}


/// Delegate the call to Gradient<Impl, Float>(impl)
template <class Impl>
auto gradient (const Impl& impl)
{
    return Gradient<Impl>(impl);
}

/** @brief Delegate the call to FunctionGradient where we have both the function and gradients separated
 * 
 *  @param func A functor having <tt>Float Func::operator()(const Vec&)</tt>
 *  @param grad A functor having either <tt>Vec Grad::operator()(const Vec&)</tt> or 
 *              <tt>void Grad::operator()(const Vec&, Vec&)</tt>
*/
template <class Func, class Grad>
auto functionGradient (const Func& func, const Grad& grad)
{
    return FunctionGradient<Func, Grad>(func, grad);
}

/** @brief Delegate the call to FunctionGradient where we have both the function and gradients in a single functor
 * 
 *  @param impl a functor having either <tt>Float operator()(const Vec&, Vec&)</tt> or
 *         <tt>std::pair<Float, Vec> operator()(const Vec&)</tt>
*/
template <class Impl>
auto functionGradient (const Impl& impl)
{
    return FunctionGradient<Impl>(impl);
}


template <class Impl>
auto hessian (const Impl& impl)
{
    return Hessian<Impl>(impl);
}
//@}

//@}


namespace poly
{

template <class V_ = Vec>
struct Function
{
    using V = V_;
    using Float = ::nlpp::impl::Scalar<V>;

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

    using GradType_1 = V (const V&);
    using GradType_2 = void (const V&, ::nlpp::impl::Plain<V>&);


    Gradient () {}

    Gradient (const std::function<GradType_1>& grad_1, ::nlpp::impl::Precedence<0>) : grad_1(grad_1)
    {
        init();
    }

    Gradient (const std::function<GradType_2>& grad_2, ::nlpp::impl::Precedence<1>) : grad_2(grad_2)
    {
        init();
    }


    template <class G>
    Gradient (const G& grad) : Gradient(grad, ::nlpp::impl::Precedence<0>{}) {}



    V gradient (const V& x)
    {
        return grad_1(x);
    }

    void gradient (const V& x, ::nlpp::impl::Plain<V>& g)
    {
        grad_2(x, g);
    }


    V operator () (const V& x)
    {
        return gradient(x);
    }

    void operator () (const V& x, ::nlpp::impl::Plain<V>& g)
    {
        gradient(x, g);
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
            grad_2 = [gradImpl = nlpp::wrap::gradient(grad_1)](const V& x, ::nlpp::impl::Plain<V>& g) mutable
            {
                gradImpl(x, g);
            };
        }
    }


    std::function<GradType_1> grad_1;
    std::function<GradType_2> grad_2;
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
    using FuncGradType_2 = Float (const V&, ::nlpp::impl::Plain<V>&);
    
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

    Float functionGradient (const V& x, ::nlpp::impl::Plain<V>& g)
    {
        return funcGrad_2(x, g);
    }


    std::pair<Float, V> operator () (const V& x)
    {
        return functionGradient(x);
    }

    Float operator () (const V& x, ::nlpp::impl::Plain<V>& g)
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
        funcGrad_2 = [funcGradImpl](const V& x, ::nlpp::impl::Plain<V>& g) mutable -> Float
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
                grad_2 = [gradImpl = nlpp::wrap::gradient(grad_1)](const V& x, ::nlpp::impl::Plain<V>& g) mutable
                {
                    gradImpl(x, g);
                };
            
            else
                grad_2 = [gradImpl = funcGrad_2](const V& x, ::nlpp::impl::Plain<V>& g) mutable
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


} // namespace wrap


} // namespace nlpp