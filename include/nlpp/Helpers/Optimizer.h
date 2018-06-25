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
                                       using TYPE::lineSearch;      \
                                       using TYPE::stop;            \
                                       using TYPE::output;



namespace nlpp
{

struct Goldstein;



/** @defgroup OptimizerGroup Optimizer
    @copydoc Optimizer.h
*/
//@{


namespace out
{

template <int, typename...>
struct GradientOptimizer;

} // namespace out


namespace stop
{

template <bool = false>
struct GradientOptimizer;

} // namespace stop


/// Parameters namespace
namespace params
{


/** @brief Base parameter class for gradient optimizers
 * 
 *  @details Define the basic variables used by any gradient based optimizer
*/
template <class LineSearch_ = Goldstein, class Stop_ = stop::GradientOptimizer<>, class Output_ = out::GradientOptimizer<0>>
struct GradientOptimizer
{
    using LineSearch = LineSearch_;
    using Stop = Stop_;
    using Output = Output_;


    /** @name
     * @brief Some basic constructors
    */
    GradientOptimizer(const LineSearch& lineSearch = LineSearch{}, const Stop& stop = Stop{}, const Output& output = Output{}) :
                      lineSearch(lineSearch), stop(stop), output(output)
    {
    }
    //@}


    LineSearch lineSearch;  ///< The line search method

    Stop stop;      ///< Stopping condition

    Output output;  ///< The output callback
};


} // namespace params



namespace out
{


template <int Level = 0, typename...>
struct GradientOptimizer;


template <>
struct GradientOptimizer<0>
{
    template <class LineSearch, class Stop, class Output, class Vec>
    void init (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
               const Eigen::MatrixBase<Vec>& x, double fx, const Eigen::MatrixBase<Vec>& gx) 
    {
    }

    template <class LineSearch, class Stop, class Output, class Vec>
    void operator() (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
               const Eigen::MatrixBase<Vec>& x, double fx, const Eigen::MatrixBase<Vec>& gx)
    {
    }

    template <class LineSearch, class Stop, class Output, class Vec>
    void finish (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
               const Eigen::MatrixBase<Vec>& x, double fx, const Eigen::MatrixBase<Vec>& gx)
    {
    }



    int iter;
    int maxIterations;

    nlpp::Vec vfx;
};


template <>
struct GradientOptimizer<1> : public GradientOptimizer<0>
{
    template <class LineSearch, class Stop, class Output, class Vec>
    void init (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
               const Eigen::MatrixBase<Vec>& x, double fx, const Eigen::MatrixBase<Vec>& gx) 
    {
        handy::print("Init\n\n", "x:", x.transpose(), "   fx:", fx, "   gx:", gx.transpose());
    }

    template <class LineSearch, class Stop, class Output, class Vec>
    void operator() (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
               const Eigen::MatrixBase<Vec>& x, double fx, const Eigen::MatrixBase<Vec>& gx)
    {
        handy::print("x:", x.transpose(), "   fx:", fx, "   gx:", gx.transpose());
    }

    template <class LineSearch, class Stop, class Output, class Vec>
    void finish (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
               const Eigen::MatrixBase<Vec>& x, double fx, const Eigen::MatrixBase<Vec>& gx)
    {
        handy::print("x:", x.transpose(), "   fx:", fx, "   gx:", gx.transpose(), "\n\nFinish");
    }
};


template <class Vec>
struct GradientOptimizer<2, Vec> : public GradientOptimizer<0>
{
    using Float = typename Vec::Scalar;

    template <class LineSearch, class Stop, class Output>
    void init (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
               const Eigen::MatrixBase<Vec>& x, Float fx, const Eigen::MatrixBase<Vec>& gx) 
    {
        vX.clear();
        vFx.clear();
        vGx.clear();

        pushBack(x, fx, gx);
    }

    template <class LineSearch, class Stop, class Output>
    void operator() (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
               const Eigen::MatrixBase<Vec>& x, Float fx, const Eigen::MatrixBase<Vec>& gx)
    {
        pushBack(x, fx, gx);
    }

    template <class LineSearch, class Stop, class Output>
    void finish (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
                 const Eigen::MatrixBase<Vec>& x, Float fx, const Eigen::MatrixBase<Vec>& gx)
    {
        pushBack(x, fx, gx);
    }


    void pushBack (const Eigen::MatrixBase<Vec>& x, Float fx, const Eigen::MatrixBase<Vec>& gx)
    {
        vX.push_back(x);
        vFx.push_back(fx);
        vGx.push_back(gx);
    }


    std::vector<Vec> vX;
    std::vector<Float> vFx;
    std::vector<Vec> vGx;
};



} // namespace out


namespace stop
{

namespace impl
{

template <class Impl>
struct GradientOptimizer
{
    GradientOptimizer(int maxIterations = 1000, double xTol = 1e-4, double fTol = 1e-4, double gTol = 1e-4) : 
                      maxIterations(maxIterations), xTol(xTol), fTol(fTol), gTol(gTol) {}

    template <class LineSearch, class Stop, class Output, class Vec>
    void init (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
               const Eigen::MatrixBase<Vec>& x, double fx, const Eigen::MatrixBase<Vec>& gx) 
    {
        x0 = x;
        fx0 = fx;
        //gx0 = gx;
    }


    template <class LineSearch, class Stop, class Output, class Vec>
    bool operator () (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
                      const Eigen::MatrixBase<Vec>& x, double fx, const Eigen::MatrixBase<Vec>& gx) 
    {
        bool xStop = (x - x0).norm() < xTol;
        bool fStop = std::abs(fx - fx0) < fTol;
        bool gStop = gx.norm() < gTol;

        x0 = x;
        fx0 = fx;
        //gx0 = gx;

        return static_cast<Impl&>(*this).stop(xStop, fStop, gStop);
    }


    Vec x0;
    double fx0;
    //Vec gx0;


    int maxIterations;      ///< Maximum number of outer iterations

	double xTol;            ///< Minimum tolerance on the norm of the input (@c x) between iterations

	double fTol;            ///< Minimum tolerance on the value of the function (@c x) between iterations

    double gTol;            ///< Minimum tolerance on the norm of the gradient (@c g) between iterations

};


} // namespace impl



template <bool Exclusive>
struct GradientOptimizer : public impl::GradientOptimizer<GradientOptimizer<Exclusive>>
{
    using impl::GradientOptimizer<GradientOptimizer<Exclusive>>::GradientOptimizer;


    bool stop (double xStop, double fStop, double gStop)
    {
        return xStop && fStop && gStop;
    }
};


template <>
struct GradientOptimizer<false> : public impl::GradientOptimizer<GradientOptimizer<false>>
{
    using impl::GradientOptimizer<GradientOptimizer<false>>::GradientOptimizer;

    bool stop (double xStop, double fStop, double gStop)
    {
        return xStop || fStop || gStop;
    }
};



} // namespace stop




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