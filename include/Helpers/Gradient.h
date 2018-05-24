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


namespace cppnlp
{

namespace wrap
{

/** @name
 *  @brief Decides whether a given function has or has not a overloaded member functions taking the given parameters
*/
//@{
HAS_OVERLOADED_FUNC(operator(), HasOp);

HAS_OVERLOADED_FUNC(function, HasFunc);

HAS_OVERLOADED_FUNC(gradient, HasGrad);
//@}


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
template <class Impl, typename Float = ::cppnlp::types::Float>
struct Gradient : public Impl
{
    /// Single constructor, delegated to Impl
    Gradient (const Impl& impl = Impl{}) : Impl(impl) {}


    /** @name
     *  @brief Overloads for @c operator() wrapping the actual implementation (@c Impl::operator())
     * 
     *  @tparam Vec A vector or matrix type represented inheriting from Eigen::DenseBase<Vec>
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
    template <class Vec, std::enable_if_t<HasOp<Impl, const Vec&>::value, int> = 0>
    auto operator () (const Eigen::MatrixBase<Vec>& x)
    {
        return Impl::operator()(x);
    }

    /** @brief Calls @a Impl::operator()(x, g), creating a vector or matrix @c g with the same dimensions as @c x.
     *         For gradient wrapping and not for function/gradient wrapping.
     *  @note Requirements:
     *        - auto Impl::operator()(const Vec&, Vec&) must be defined and returns nothing
    */
    template <class Vec, std::enable_if_t<HasOp<Impl, const Vec&, Vec&>::value && !HasOp<Impl, const Vec&>::value &&
                                          std::is_same<std::result_of_t<Impl(const Vec&, Vec&)>, void>::value, int> = 0>
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
    template <class Vec, std::enable_if_t<HasOp<Impl, const Vec&, Vec&>::value && !HasOp<Impl, const Vec&>::value &&
                                          !std::is_same<std::result_of_t<Impl(const Vec&, Vec&)>, void>::value, int> = 0>
    auto operator () (const Eigen::MatrixBase<Vec>& x)
    {
        Eigen::Matrix<Float, Vec::RowsAtCompileTime, Vec::ColsAtCompileTime> g(x.rows(), x.cols());

        auto f = Impl::operator()(x, g);
        
        return std::make_pair(f, g);
    }
    

    /** @brief Set the value of the reference @c g to the return of @c Impl::operator()(x) when it returns the gradient.
     *         For gradient only wrapping, when the return is the actual gradient.
     *  @note Requirements:
     *        - auto Impl::operator()(const Vec&) must be defined and returns the gradient of a matrix type
    */
    template <class Vec, std::enable_if_t<HasOp<Impl, const Vec&>::value && 
                                          ::cppnlp::impl::isMat<std::result_of_t<Impl(const Vec&)>>, int> = 0>
    void operator () (const Eigen::MatrixBase<Vec>& x, Eigen::MatrixBase<Vec>& g)
    {
        g = Impl::operator()(x);
    }

    /** @brief Set the value of the reference @c g to the return of @c Impl::operator()(x) when it returns the gradient.
     *         For function/gradient only wrapping, when the return is the function value and the gradient is passed by reference.
     *  @note Requirements:
     *        - auto Impl::operator()(const Vec&) must be defined and returns the gradient of a scalar type
    */
    template <class Vec, std::enable_if_t<HasOp<Impl, const Vec&>::value && 
                                          ::cppnlp::impl::isScalar<std::result_of_t<Impl(const Vec&)>>, int> = 0>
    auto operator () (const Eigen::MatrixBase<Vec>& x, Eigen::MatrixBase<Vec>& g)
    {
        auto [f, g_] = Impl::operator()(x);

        g = g_;

        return f;
    }

    /** @Simply delegate the call to Impl::operator()(x, g), where the gradient value will be saved in @c g, and
     *          returns the function value.
     * 
     *  @note Requirements:
     *        - auto Impl::operator()(const Vec&, Vec&) must be defined
    */
    template <class Vec, std::enable_if_t<HasOp<Impl, const Vec&, Vec&>::value && 
                                          !HasOp<Impl, const Vec&>::value, int> = 0>
    auto operator () (const Eigen::MatrixBase<Vec>& x, Eigen::MatrixBase<Vec>& g)
    {
        return Impl::operator()(x, static_cast<Vec&>(g));
    }
    //@}
};






/** @name 
 *  @brief The uniform interface wrapper for function/gradient functors
 *  
 *  @details This class provides the uniform interface where, given an user defined function/gradient functor or
 *           both function and gradient functors separated, you can call for function/gradient, function only pr
 *           gradient only, always avoiding to execute additional function calls when possible. 
 *  
 *  
 *  
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
struct FunctionGradient<Func, Grad> : public Func, public Gradient<Grad>
{
    /// Single constructor, delegated to Func and Grad
    FunctionGradient (const Func& f = Func{}, const Grad& g = Grad{}) : Func(f), Gradient<Grad>(g) {}
    
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
        return std::make_pair(Func::operator()(x), Gradient<Grad>::operator()(x));
    }

    template <class Vec>
    auto operator () (const Eigen::MatrixBase<Vec>& x, Eigen::MatrixBase<Vec>& g)
    {
        Gradient<Grad>::operator()(x, g);

        return Func::operator()(x);
    }
    //@}


    /** @name
     *  @brief Delegation for the base implementation of the function and gradient calls
     * 
     *  @param x The vector for which we will be evaluating both function and gradient
     *  @param g The reference where we are going to store the gradient
    */
    //@{
    template <class Vec>
    auto function (const Eigen::MatrixBase<Vec>& x)
    {
        return Func::operator()(x.eval());
    }

    template <class Vec>
    auto gradient (const Eigen::MatrixBase<Vec>& x)
    {
        return Gradient<Grad>::operator()(x.eval());
    }

    template <class Vec>
    void gradient (const Eigen::MatrixBase<Vec>& x, Eigen::MatrixBase<Vec>& g)
    {
        Gradient<Grad>::operator()(x.eval(), g);
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
    template <typename Float, int Rows, int Cols>
    auto operator () (const Eigen::Matrix<Float, Rows, Cols>& x)
    {
        return Gradient<FuncGrad>::operator()(x);
    }

    template <class Vec>
    auto operator () (const Eigen::MatrixBase<Vec>& x)
    {
        return operator()(x.eval());
    }

    template <typename Float, int Rows, int Cols>
    auto operator () (const Eigen::Matrix<Float, Rows, Cols>& x, Eigen::Matrix<Float, Rows, Cols>& g)
    {
        return Gradient<FuncGrad>::operator()(x, g);
    }

    template <class Vec>
    auto operator () (const Eigen::MatrixBase<Vec>& x, Eigen::MatrixBase<Vec>& g)
    {
        return operator()(x.eval(), g);
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


/** @name 
 *  @brief Functions used only to delegate the call with automatic type deduction
*/
//@{

/// Delegate the call to Gradient<impl, Float>(impl)
template <class Impl, typename Float = ::cppnlp::types::Float>
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

} // namespace wrap


} // namespace cppnlp