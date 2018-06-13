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




/// Base parameter definitions
#define CPPOPT_USING_PARAMS(TYPE, ...) using TYPE = __VA_ARGS__;    \
                                       using TYPE::maxIterations;   \
                                       using TYPE::xTol;            \
                                       using TYPE::fTol;            \
                                       using TYPE::gTol;            \
                                       using TYPE::lineSearch;      \
                                       using TYPE::output;




namespace nlpp
{

struct Goldstein;


namespace out
{

template <int>
struct GradientOptimizer;

} // namespace out


/// Parameters namespace
namespace params
{


/** @defgroup OptimizerGroup Optimizer
    @copydoc Optimizer.h
*/
//@{


/** @brief Base parameter class for gradient optimizers
 * 
 *  @details Define the basic variables used by any gradient based optimizer
*/
template <class LineSearch = Goldstein, class Output = out::GradientOptimizer<0>>
struct GradientOptimizer
{
    /** @name
     * @brief Some basic constructors
    */
    GradientOptimizer(int maxIterations = 1000, double xTol = 1e-4, double fTol = 1e-4, 
                      double gTol = 1e-4, const LineSearch& lineSearch = LineSearch()) :
                      maxIterations(maxIterations), xTol(xTol), fTol(fTol), gTol(gTol), lineSearch(lineSearch) {}

    GradientOptimizer(const LineSearch& lineSearch,int maxIterations = 1000, double xTol = 1e-4, double fTol = 1e-4, double gTol = 1e-4) :
                      maxIterations(maxIterations), xTol(xTol), fTol(fTol), gTol(gTol), lineSearch(lineSearch) {}

    GradientOptimizer(const LineSearch& lineSearch, const Output& output, 
                        int maxIterations = 1000, double xTol = 1e-4, double fTol = 1e-4, double gTol = 1e-4) :
                        maxIterations(maxIterations), xTol(xTol), fTol(fTol), gTol(gTol), lineSearch(lineSearch), output(output) {}
    //@}



    int maxIterations;      ///< Maximum number of outer iterations

	double xTol;            ///< Minimum tolerance on the norm of the input (@c x) between iterations

	double fTol;            ///< Minimum tolerance on the value of the function (@c x) between iterations

    double gTol;            ///< Minimum tolerance on the norm of the gradient (@c g) between iterations


    LineSearch lineSearch;  ///< The line search method to be used

    Output output;  ///< The output callback
};


} // namespace params



namespace out
{


template <int Level>
struct GradientOptimizer;


template <>
struct GradientOptimizer<0>
{
    template <class LineSearch, class Output>
    void init (const params::GradientOptimizer<LineSearch, Output>& optimizer)
    {
    }

    template <class LineSearch, class Output>
    void init (const params::GradientOptimizer<LineSearch, Output>& optimizer, double fx) 
    {
    }

    template <class LineSearch, class Output>
    void operator () (const params::GradientOptimizer<LineSearch, Output>& optimizer, double fx)
    {
    }


    template <class LineSearch, class Output>
    void finish (const params::GradientOptimizer<LineSearch, Output>& optimizer, double fx)
    {
    }



    int iter;
    int maxIterations;

    nlpp::Vec vfx;
};


template <>
struct GradientOptimizer<1> : public GradientOptimizer<0>
{
    template <class LineSearch, class Output>
    void init (const params::GradientOptimizer<LineSearch, Output>& optimizer, double fx) 
    {
        handy::print("init fx: ", fx);
    }

    template <class LineSearch, class Output>
    void operator () (const params::GradientOptimizer<LineSearch, Output>& optimizer, double fx)
    {
        handy::print("fx: ", fx);        
    }


    template <class LineSearch, class Output>
    void finish (const params::GradientOptimizer<LineSearch, Output>& optimizer, double fx)
    {
        handy::print("finish fx: ", fx);        
    }
};



} // namespace out



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
template <class Impl, class Parameters = params::GradientOptimizer<>>
struct GradientOptimizer : public Parameters
{
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
        return operator()(func, fd::gradient(func), x, std::forward<Args>(args)...);
    }
    //@}
};

//@}

} // namespace nlpp