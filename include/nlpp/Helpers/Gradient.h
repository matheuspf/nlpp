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

#include "Helpers.h"


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
 
template <class T, class V = Vec>
struct IsFunction
{
    template <class U = T>
    static constexpr bool impl (std::decay_t<decltype(std::declval<U>()(V()), void())>*)
    { 
        return std::is_floating_point<decltype(std::declval<U>()(V()))>::value;
    }

    static constexpr bool impl (...)
    {
        return false;
    }
    

    enum{ value = impl(nullptr) };
};

template <class T, class V = Vec>
struct IsGradient
{
    static V ref;

    template <class U = T>
    static constexpr bool impl (decltype(std::declval<U>()(V()), void())*)
    { 
        return ::nlpp::impl::IsMat<decltype(std::declval<U>()(V()))>::value;
    }

    template <class U = T>
    static constexpr bool impl (decltype(std::declval<U>()(V(), ref), int())*)
    { 
        return std::is_same<decltype(std::declval<U>()(V(), ref)), void>::value;
    }

    static constexpr bool impl (...)
    {
        return false;
    }
    

    enum{ value = impl((int*)0) };
};

template <class T, class V>
V IsGradient<T, V>::ref;


template <class T, class V = Vec>
struct IsFunctionGradient
{
    static V ref;


    template <class U>
    struct IsPair
    {
        enum { value = false };
    };

    template <typename U, typename W>
    struct IsPair<std::pair<U, W>>
    {
        enum { value = true };
    };


    template <class U = T>
    static constexpr bool impl (decltype(std::declval<U>()(V()), void())*)
    { 
        return IsPair<decltype(std::declval<U>()(V()))>::value;
        
        // return IsPair<decltype(std::declval<U>()(V()))>::value &&
        //        std::is_floating_point<decltype(std::get<0>(std::declval<U>()(V())))>::value &&
        //        impl::IsMat<decltype(std::get<1>(std::declval<U>()(V())))>::value;
    }

    template <class U = T>
    static constexpr bool impl (decltype(std::declval<U>()(V(), ref), int())*)
    { 
        return std::is_floating_point<decltype(std::declval<U>()(V(), ref))>::value;
    }

    static constexpr bool impl (...)
    {
        return false;
    }
    

    enum{ value = impl((int*)0) };
};

template <class T, class V>
V IsFunctionGradient<T, V>::ref;
//@}



namespace impl
{

/** @brief Function wrapping for user uniform defined function calculation
 * 
*/
template <class Impl, typename Float = ::nlpp::types::Float>
struct Function : public Impl
{
    Function (const Impl& impl = Impl{}) : Impl(impl) {}

    
};

} // namespace impl


template <class Impl, typename Float = ::nlpp::types::Float>
using Function = std::conditional_t<handy::IsSpecialization<Impl, impl::Function>::value,
                                    Impl,
                                    std::conditional_t<IsFunction<Impl>::value || IsFunctionGradient<Impl>::value,
                                                       impl::Function<Impl, Float>,
                                                       void>>;






/** @brief Gradient wrapper for user defined gradient or function/gradient calculation
 * 
 *  @details Given an user defined gradient or function/gradient functor, given by Impl, it returns an object that
 *           wraps the call to the functor, avoiding an extra vector creation when possible. Example:
 * 
 *  
 *  @snippet Helpers/Gradient.cpp Gradient snippet
 * 
 * 
 *  @tparam Impl The actual implementation of the gradient function
 *  @tparam Float the base scalar floating point type for creating vector/arrays
 * 
 *  @note Requirements:
 *        - Only one of the following must be defined:
 *            - auto Impl::operator()(const Vec&)
 *            - auto Impl::operator()(const Vec&, Vec&)
*/
template <class Impl, typename Float = ::nlpp::types::Float>
struct Gradient : public Impl
{
    /// Single constructor, delegated to Impl
    Gradient (const Impl& impl = Impl{}) : Impl(impl) {}


    /** @name
     *  @brief Overloads for @c operator() wrapping the actual implementation (@c Impl::operator())
     * 
     *  @tparam Vec A vector or matrix type inheriting from Eigen::DenseBase<Vec>
     *  @param x The vector or matrix where the actual function will be called at
     *  @param g A reference to a vector, where the gradient of the actual implementation will be saved
     * 
     *  @note Requirements:
     *        - x.rows() == g.rows()
     *        - x.cols() == g.cols()
    */
    //@{
    
    /** @brief Direcly calls @a Impl::operator()(x)
     *  @note Requirements:
     *        - auto Impl::operator()(const Vec&) must be defined
    */
    template <class Vec, class Impl_ = Impl,
              std::enable_if_t<HasOp<Impl_, const Vec&>::value, int> = 0>

    auto operator () (const Eigen::MatrixBase<Vec>& x)
    {
        return Impl::operator()(x);
    }

    /** @brief Calls @a Impl::operator()(x, g), creating a vector or matrix @c g with the same dimensions as @c x.
     *         For gradient wrapping and not for function/gradient wrapping.
     *  @note Requirements:
     *        - auto Impl::operator()(const Vec&, Vec&) must be defined and returns nothing
    */
    template <class Vec, class Impl_ = Impl, 
              std::enable_if_t<IsGradient<Impl_, Vec>::value && 
                               HasOp<Impl_, const Vec&, Vec&>::value &&
                               !HasOp<Impl_, const Vec&>::value, int> = 0>

    auto operator () (const Eigen::MatrixBase<Vec>& x)
    {
        Eigen::Matrix<Float, Vec::RowsAtCompileTime, Vec::ColsAtCompileTime> g(x.rows(), x.cols());

        Impl::operator()(x, g);
        
        return g;
    }

    /** @brief Calls @a Impl::operator()(x, g), creating a vector or matrix @c g with the same dimensions as @c x.
     *         For function/gradient wrapping and not for gradient only wrapping.
     *  @note Requirements:
     *        - auto Impl::operator()(const Vec&, Vec&) must be defined and returns something (it does not need to be a scalar)
    */
    template <class Vec, class Impl_ = Impl,
              std::enable_if_t<IsFunctionGradient<Impl_, Vec>::value && 
                               HasOp<Impl_, const Vec&, Vec&>::value &&
                               !HasOp<Impl_, const Vec&>::value, int> = 0>

    auto operator () (const Eigen::MatrixBase<Vec>& x)
    {
        Eigen::Matrix<Float, Vec::RowsAtCompileTime, Vec::ColsAtCompileTime> g(x.rows(), x.cols());

        auto f = Impl::operator()(x, g);
        
        return std::make_pair(f, g);
    }



    /** @Simply delegate the call to Impl::operator()(x, g), where the gradient value will be saved in @c g, and
     *          returns the function value.
     * 
     *  @note Requirements:
     *        - auto Impl::operator()(const Vec&, Vec&) must be defined
    */
    template <class Vec, class Impl_ = Impl,
              std::enable_if_t<HasOp<Impl_, const Vec&, Vec&>::value, int> = 0>
              
    auto operator () (const Eigen::MatrixBase<Vec>& x, Eigen::MatrixBase<Vec>& g)
    {
        return Impl::operator()(x, static_cast<Vec&>(g));
    }

    /** @brief Set the value of the reference @c g to the return of @c Impl::operator()(x) when it returns the gradient.
     *         For gradient only wrapping, when the return is the actual gradient.
     *  @note Requirements:
     *        - auto Impl::operator()(const Vec&) must be defined and returns the gradient of a matrix type
    */
    template <class Vec, class Impl_ = Impl, 
              std::enable_if_t<HasOp<Impl_, const Vec&>::value &&
                               IsGradient<Impl_, Vec>::value &&
                               !HasOp<Impl_, const Vec&, Vec&>::value, int> = 0>

    void operator () (const Eigen::MatrixBase<Vec>& x, Eigen::MatrixBase<Vec>& g)
    {
        g = Impl::operator()(x);
    }

    /** @brief Set the value of the reference @c g to the return of @c Impl::operator()(x) when it returns the gradient.
     *         For function/gradient only wrapping, when the return is the function value and the gradient is passed by reference.
     *  @note Requirements:
     *        - auto Impl::operator()(const Vec&) must be defined and returns the gradient of a scalar type
    */
    template <class Vec, class Impl_ = Impl,
              std::enable_if_t<HasOp<Impl_, const Vec&>::value &&
                               IsFunctionGradient<Impl_, Vec>::value &&
                               !HasOp<Impl_, const Vec&, Vec&>::value, int> = 0>

    auto operator () (const Eigen::MatrixBase<Vec>& x, Eigen::MatrixBase<Vec>& g)
    {
        Float f;
        
        std::tie(f, g) = Impl::operator()(x);

        return f;
    }
    //@}


    template <class Vec>
    auto gradient (const Eigen::MatrixBase<Vec>& x)
    {
        return operator()(x.eval());
    }

    template <class Vec>
    void gradient (const Eigen::MatrixBase<Vec>& x, Eigen::MatrixBase<Vec>& g)
    {
        operator()(x.eval(), g);
    }
};



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
    /// Single constructor, delegated to Func and Grad
    FunctionGradient (const Func& f = Func{}, const Grad& g = Grad{}) : Function<Func>(f), Gradient<Grad>(g) {}
    
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
    template <class Vec>
    auto operator () (const Vec& x)
    {
        return std::make_pair(function(x), gradient(x));
    }

    template <class X, class G>
    auto operator () (const Eigen::MatrixBase<X>& x, Eigen::MatrixBase<G>& g)
    {
        gradient(x, g);

        return function(x);
    }
    //@}
};


/** @brief Specialization for when both function and gradients are given in a single functor
 * 
 *  @tparam FuncGrad A functor having either <tt>std::pair<Float, Vec> Func::operator()(const Vec&)</tt> or 
 *          <tt>Float Func::operator()(const Vec&, Vec&)</tt> defined.
 * 
 *  @note This class inherits @c FuncGrad, wrapping it first into @c Gradient<FuncGrad>
*/
template <class FuncGrad>
struct FunctionGradient<FuncGrad> : public Gradient<FuncGrad>
{
    /// A single constructor, delegating the call to @c FuncGrad
    FunctionGradient (const FuncGrad& fg = FuncGrad{}) : Gradient<FuncGrad>(fg) {}


    /** @name
     *  @brief Here, we simply delegate the call to the actual implementation.
     * 
     *  @param x The vector for which we will be evaluating both function and gradient
     *  @param g The reference where we are going to store the gradient
    */
    //@{
    template <class X>
    auto operator () (const Eigen::MatrixBase<X>& x)
    {
        return Gradient<FuncGrad>::operator()(x.eval());
    }

    template <class X, class G>
    auto operator () (const Eigen::MatrixBase<X>& x, Eigen::MatrixBase<G>& g)
    {
        return Gradient<FuncGrad>::operator()(x.eval(), g);
    }
    //@}


    /** @name
     *  @brief Calling function and gradient separatelly
     * 
     *  @details We check first if there is a member function with name "function" for function call,
     *           and "gradient", for gradient calls, in the @c Gradient<FuncGrad> functor. If it exists,
     *           we call it. 
     * 
     *           Otherwise, we have no option other than calculating both function and gradient (by calling
     *           Gradient<FuncGrad>::operator()) and only returning the desired part.
     *  
     *  @param x The vector for which we will be evaluating both function and gradient
     *  @param g The reference where we are going to store the gradient
    */
    //@{
    template <class Vec, std::enable_if_t<HasFunc<FuncGrad, const Vec&>::value, int> = 0>
    auto function (const Eigen::MatrixBase<Vec>& x)
    {
        return Gradient<FuncGrad>::function(x.eval());
    }

    template <class Vec, std::enable_if_t<!HasFunc<FuncGrad, const Vec&>::value, int> = 0>
    auto function (const Eigen::MatrixBase<Vec>& x)
    {
        return std::get<0>(Gradient<FuncGrad>::operator()(x.eval()));
    }


    template <class Vec, std::enable_if_t<HasGrad<FuncGrad, const Vec&>::value, int> = 0>
    auto gradient (const Eigen::MatrixBase<Vec>& x)
    {
        return Gradient<FuncGrad>::gradient(x.eval());
    }

    template <class Vec, std::enable_if_t<!HasGrad<FuncGrad, const Vec&>::value, int> = 0>
    auto gradient (const Eigen::MatrixBase<Vec>& x)
    {
        return std::get<1>(Gradient<FuncGrad>::operator()(x.eval()));
    }


    template <class Vec, std::enable_if_t<HasGrad<FuncGrad, const Vec&, Vec&>::value, int> = 0>
    void gradient (const Eigen::MatrixBase<Vec>& x, Eigen::MatrixBase<Vec>& g)
    {
        Gradient<FuncGrad>::gradient(x.eval(), g);
    }

    template <class Vec, std::enable_if_t<!HasGrad<FuncGrad, const Vec&, Vec&>::value, int> = 0>
    void gradient (const Eigen::MatrixBase<Vec>& x, Eigen::MatrixBase<Vec>& g)
    {
        Gradient<FuncGrad>::operator()(x.eval(), g);
    }
    //@}
};
//@}


} // namespace impl



/** @brief Alias for impl::FunctionGradient
 *  @details There are three conditions:
 *           - If @c Grad is not given and
 *              - If @c Func is already a impl::FunctionGradient, simply set the result to itself (to avoid multiple wrapping)
 *              - Otherwise set the result to impl::FunctionGradient<Func>
 *           - Otherwise set the result to impl::FunctionGradient<Func, Grad>
*/
template <class Func, class Grad = void>
using FunctionGradient = std::conditional_t<std::is_same<Grad, void>::value, 
                                            std::conditional_t<handy::IsSpecialization<Func, impl::FunctionGradient>::value, 
                                                               Func, 
                                                               impl::FunctionGradient<Func>>,
                                            impl::FunctionGradient<Func, Grad>>;





/** @name 
 *  @brief Functions used only to delegate the call with automatic type deduction
*/
//@{

/// Delegate the call to Function<Impl, Float>(impl)
template <class Impl, typename Float = ::nlpp::types::Float>
auto function (const Impl& impl)
{
    return Function<Impl, Float>(impl);
}


/// Delegate the call to Gradient<Impl, Float>(impl)
template <class Impl, typename Float = ::nlpp::types::Float>
auto gradient (const Impl& impl)
{
    return Gradient<Impl, Float>(impl);
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
 *  @param funcGrad a functor having either <tt>Float operator()(const Vec&, Vec&)</tt> or
 *         <tt>std::pair<Float, Vec> operator()(const Vec&)</tt>
*/
template <class FuncGrad>
auto functionGradient (const FuncGrad& funcGrad)
{
    return FunctionGradient<FuncGrad>(funcGrad);
}
//@}

//@}

} // namespace wrap


} // namespace nlpp