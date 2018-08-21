/** @file
 *  @brief Optimizer base CRTP class
 * 
 *  @details Expose some default interface, wrapping function and gradient calls before delegating to
 *           the actual implementation. Also defines base parameter class for optimizers.
*/

#pragma once



//#include "../LineSearch/Goldstein/Goldstein.h"

#include "Helpers.h"

#include "FiniteDifference.h"

#include "Gradient.h"

#include "Output.h"

#include "Stop.h"

#include "Parameters.h"




/// Base parameter definitions
#define CPPOPT_USING_PARAMS(TYPE, ...) using TYPE = __VA_ARGS__;    \
                                       using TYPE::lineSearch;      \
                                       using TYPE::stop;            \
                                       using TYPE::output;



namespace nlpp
{

/** @defgroup OptimizerGroup Optimizer
    @copydoc Optimizer.h
*/
//@{


/** @brief Base class for gradient based optimizers
 *  
 *  @details This class simply inherits base parameters and delegates the call to @c Impl by first wrapping the 
 *           given arguments. 
 * 
 *  @tparam Impl The actual implementation, inheriting from GradientOptimizer<Impl, Parameters>
 *  @tparam Parameters The base parameters for @c Impl, given as argument by @c Impl
 * 
 *  @note We inherit from the parameter classes from here, so we create a single inheritance chain, not having to 
 *        resort to multiple inheritance.
*/
template <class Impl, class Parameters_ = params::GradientOptimizer<>>
struct GradientOptimizer : public Parameters_
{
    CPPOPT_USING_PARAMS(Parameters, Parameters_);
    using Parameters::Parameters;

    /// Simply delegate the call to the base parameters class
    GradientOptimizer(const Parameters& params = Parameters()) : Parameters(params) {}


    /** @name
     *  @brief Ensure uniform interface, wrapping arguments before delegation
     * 
     *  @details Given a function OR function and gradient OR function/gradient, wrap the calls accordingly before delegating
     *           the call to the @c optimize function of the actual implementation
     * 
     *  @tparam Function A scalar function functor
     *  @tparam Gradient A gradient function functor
     *  @tparam FunctionGradient A function/gradient functor
     *  @tparam Vec The Eigen::MatrixBase input argument
     *  @tparam Args... Any additional parameter to be passed to the optimizer
    */
    //@{

    /// When the function and gradient functors are given separatelly
    template <class Function, class Gradient, class Vec, typename... Args>
            //   std::enable_if_t<wrap::IsFunction<Function, Vec>::value && wrap::IsGradient<Gradient, Vec>::value, int> = 0>
    Vec operator () (const Function& function, const Gradient& gradient, const Eigen::MatrixBase<Vec>& x, Args&&... args)
    {
        return static_cast<Impl&>(*this).optimize(wrap::functionGradient(function, gradient), x.eval(), std::forward<Args>(args)...);
    }

    /// When the function and gradient functors are given in a single class
    template <class FunctionGradient, class Vec, typename... Args,
              std::enable_if_t<wrap::IsFunctionGradient<FunctionGradient, Vec>::value, int> = 0>
    Vec operator () (const FunctionGradient& funcGrad, const Eigen::MatrixBase<Vec>& x, Args&&... args)
    {
        return static_cast<Impl&>(*this).optimize(wrap::functionGradient(funcGrad), x.eval(), std::forward<Args>(args)...);
    }

    /// When only the function functor is given - in this case, we use finite difference to estimate the gradient
    template <class Function, class Vec, typename... Args, 
              std::enable_if_t<wrap::IsFunction<Function, Vec>::value, int> = 0>
    Vec operator () (const Function& func, const Eigen::MatrixBase<Vec>& x, Args&&... args)
    {
        return operator()(func, fd::gradient(func), x.eval(), std::forward<Args>(args)...);
    }
    //@}
};

//@}

} // namespace nlpp