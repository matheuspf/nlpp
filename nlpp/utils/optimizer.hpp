/** @file
 *  @brief Optimizer base CRTP class
 * 
 *  @details Expose some default interface, wrapping function and gradient calls before delegating to
 *           the actual implementation. Also defines base parameter class for optimizers.
*/

#pragma once



//#include "../LineSearch/Goldstein/Goldstein.h"

#include "helpers/helpers.hpp"
#include "utils/finiteDifference.hpp"
#include "utils/wrappers.hpp"
#include "utils/output.hpp"
#include "utils/stop.hpp"

#include "../LineSearch/LineSearch.h"


/// Base parameter definitions
#define NLPP_USING_OPTIMIZER(TYPE, ...) using TYPE = __VA_ARGS__;    \
                                        using TYPE::stop;            \
                                        using TYPE::output;          \
                                        using TYPE::TYPE;

#define NLPP_USING_GRADIENT_OPTIMIZER(TYPE, ...) NLPP_USING_OPTIMIZER(TYPE, __VA_ARGS__)

#define NLPP_USING_LINESEARCH_OPTIMIZER(TYPE, ...) NLPP_USING_GRADIENT_OPTIMIZER(TYPE, __VA_ARGS__) \
                                                   using TYPE::lineSearch;


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
 *  @tparam Parameters_ The base parameters for @c Impl, given as argument by @c Impl
 * 
 *  @note We inherit from the parameter classes here, so we create a single inheritance chain, not having to 
 *        resort to multiple inheritance. Also, this way we can have easy access to any member of the parameters class
*/

// class LineSearch = StrongWolfe<>, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>

template <class Impl, class Stop = stop::Optimizer<>, class Output = out::Optimizer<>>
struct Optimizer
{
    // Optimizer (const Optimizer&) = default;
    // Optimizer (Optimizer&&) = default;

    Optimizer (const Stop& stop = Stop{}, const Output& output = Output{}) : stop(stop), output(output)
    {
    }

    void initialize ()
    {
        stop.initialize();
        output.initialize();
    }

    template <class Function, class V, typename... Args>
    impl::Plain<V> operator () (const Function& function, const Eigen::MatrixBase<V>& x, Args&&... args)
    {
        return static_cast<Impl&>(*this).optimize(wrap::function(function), x.eval(), std::forward<Args>(args)...);
    }


    Stop stop;      ///< Stopping condition

    Output output;  ///< The output callback
};


template <class Impl, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
struct GradientOptimizer : public Optimizer<Impl, Stop, Output>
{
    using Base = Optimizer<Impl, Stop, Output>;
    using Base::Base;

    /** @name
     *  @brief Ensure uniform interface, wrapping arguments before delegation
     * 
     *  @details Given a function OR function and gradient OR function/gradient, wrap the calls accordingly before delegating
     *           the call to the @c optimize function of the actual implementation. If a function only is given, a default finite 
     *           gradient estimation is used. The wrapping interface takes care of the complexity, so we only need to delegate the 
     *           given parameters.
     * 
     *  @tparam Function A scalar function functor
     *  @tparam Gradient A gradient function functor
     *  @tparam FunctionGradient A function/gradient functor
     *  @tparam V A Eigen object input argument, which could be also an expression (it is evaluated before the call to the optimize function)
     *  @tparam Args... Any additional parameter to be passed to the optimizer
    */
    //@{
    template <class Function, class Gradient, class V, typename... Args>
    impl::Plain<V> operator () (const Function& function, const Gradient& gradient, const Eigen::MatrixBase<V>& x, Args&&... args)
    {
        return static_cast<Impl&>(*this).optimize(wrap::functionGradient(function, gradient), x.eval(), std::forward<Args>(args)...);
    }

    template <class Function, class V, typename... Args>
    impl::Plain<V> operator () (const Function& function, const Eigen::MatrixBase<V>& x, Args&&... args)
    {
        return static_cast<Impl&>(*this).optimize(wrap::functionGradient(function), x.eval(), std::forward<Args>(args)...);
    }

    template <class Function, class Gradient, class Hessian, class V, typename... Args>
    impl::Plain<V> operator () (const Function& function, const Gradient& gradient, const Hessian& hessian, const Eigen::MatrixBase<V>& x, Args&&... args)
    {
        return static_cast<Impl&>(*this).optimize(wrap::functionGradient(function, gradient), wrap::hessian(hessian), x.eval(), std::forward<Args>(args)...);
    }
    //@}
};

template <class Impl, class LineSearch = StrongWolfe<>, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
struct LineSearchOptimizer : public GradientOptimizer<Impl, Stop, Output>
{
    using Base = GradientOptimizer<Impl, Stop, Output>;
    using Base::Base;

    LineSearchOptimizer (const LineSearch& lineSearch = LineSearch{}, const Stop& stop = Stop{}, const Output& output = Output{}) :
                         lineSearch(lineSearch), Base(stop, output)
    {
    }

    void initialize ()
    {
        Base::initialize();
        lineSearch.initialize();
    }

    LineSearch lineSearch;
};


//@}


namespace poly
{

template <class V = ::nlpp::Vec>
struct Optimizer // : public CloneBase<Optimizer<V>>
{
    Optimizer (const out::Optimizer_<V>& output = out::Optimizer_<V>{},
               const stop::Optimizer_<V>& stop = stop::Optimizer_<V>{}) : output(output), stop(stop)
    {
    }

    virtual void init ()
    {
        output.initialize();
        stop.initialize();
    }

    virtual V optimize (::nlpp::wrap::poly::Function<V>, V) { return V{}; }


    template <class Function, class U, typename... Args>
    ::nlpp::impl::Plain<V> operator () (const Function& function, const Eigen::MatrixBase<U>& x, Args&&... args)
    {
        return optimize(::nlpp::wrap::poly::Function<::nlpp::impl::Plain<U>>(function), x.eval(), std::forward<Args>(args)...);
    }

    template <class Function, class T, int R, int C>
    Eigen::Matrix<T, R, C> operator () (const Function& function, const Eigen::Matrix<T, R, C>& x)
    {
        return optimize(::nlpp::wrap::poly::Function<Eigen::Matrix<T, R, C>>(function), x);
    }
 

    out::Optimizer_<V> output;
    stop::Optimizer_<V> stop;
};


template <class V = ::nlpp::Vec>
struct GradientOptimizer
{
    GradientOptimizer (const out::GradientOptimizer_<V>& output = out::GradientOptimizer_<V>{},
               const stop::GradientOptimizer_<V>& stop = stop::GradientOptimizer_<V>{}) : output(output), stop(stop)
    {
    }

    virtual void init ()
    {
        output.initialize();
        stop.initialize();
    }

    virtual V optimize (::nlpp::wrap::poly::FunctionGradient<V>, V) = 0;
    // virtual V optimize (::nlpp::wrap::poly::FunctionGradient<V>, ::nlpp::wrap::poly::Hessian<V>, V) = 0;


    template <class Function, class U>
    ::nlpp::impl::Plain<V> operator () (const Function& function, const Eigen::MatrixBase<U>& x)
    {
        return optimize(::nlpp::wrap::poly::FunctionGradient<::nlpp::impl::Plain<U>>(function), x.eval());
    }
   
    template <class Function, class Gradient, class U>
    ::nlpp::impl::Plain<V> operator () (const Function& function, const Gradient& gradient, const U& x)
    {
        return optimize(::nlpp::wrap::poly::FunctionGradient<::nlpp::impl::Plain<U>>(function, gradient), x.eval());
    }

    // template <class Function, class Gradient, class Hessian, class U>
    // ::nlpp::impl::Plain<V> operator () (const Function& function, const Gradient& gradient, const Hessian& hessian, const Eigen::MatrixBase<U>& x)
    // {
    //     return optimize(::nlpp::wrap::poly::FunctionGradient<::nlpp::impl::Plain<U>>(function, gradient), ::nlpp::wrap::poly::Hessian<::nlpp::impl::Plain<U>>(hessian), x.eval());
    // }

    
    template <class Function, class T, int R, int C>
    Eigen::Matrix<T, R, C> operator () (const Function& function, const Eigen::Matrix<T, R, C>& x)
    {
        return optimize(::nlpp::wrap::poly::FunctionGradient<Eigen::Matrix<T, R, C>>(function), x);
    }
    
    template <class Function, class Gradient, class T, int R, int C>
    Eigen::Matrix<T, R, C> operator () (const Function& function, const Gradient& gradient, const Eigen::Matrix<T, R, C>& x)
    {
        return optimize(::nlpp::wrap::poly::FunctionGradient<Eigen::Matrix<T, R, C>>(function, gradient), x);
    }

    // template <class Function, class Gradient, class Hessian, class T, int R, int C>
    // Eigen::Matrix<T, R, C> operator () (const Function& function, const Gradient& gradient, const Hessian& hessian, const Eigen::Matrix<T, R, C>& x)
    // {
    //     return optimize(::nlpp::wrap::poly::FunctionGradient<Eigen::Matrix<T, R, C>>(function, gradient), ::nlpp::wrap::poly::Hessian<Eigen::Matrix<T, R, C>>(hessian), x);
    // }


    out::GradientOptimizer_<V> output;
    stop::GradientOptimizer_<V> stop;
};

template <class V = ::nlpp::Vec>
struct LineSearchOptimizer : public GradientOptimizer<V>
{
    using Base = GradientOptimizer<V>;
    using Base::output;
    using Base::stop;

    virtual void init ()
    {
        Base::init();
        lineSearch.initialize();
    }
    
    LineSearchOptimizer (const LineSearch_<V>& lineSearch = LineSearch_<V>{},
                         const out::GradientOptimizer_<V>& output = out::GradientOptimizer_<V>{},
                         const stop::GradientOptimizer_<V>& stop = stop::GradientOptimizer_<V>{}) :
                         lineSearch(lineSearch), Base(output, stop)
    {
    }

    LineSearch_<V> lineSearch;
};

} // namespace poly

} // namespace nlpp