/** @file
 *  @brief Optimizer base CRTP class
 * 
 *  @details Expose some default interface, wrapping function and gradient calls before delegating to
 *           the actual implementation. Also defines base parameter class for optimizers.
*/

#pragma once



//#include "../LineSearch/Goldstein/Goldstein.h"

#include "Helpers.h"

#include "include/nlpp/Helpers/FiniteDifference.h"

#include "Wrappers.h"

#include "Output.h"

#include "Stop.h"

#include "../LineSearch/LineSearch.h"


namespace nlpp::poly
{

struct Optimizer
{
    Optimizer (const out::Optimizer_& output = out::Optimizer_{},
               const stop::Optimizer_& stop = stop::Optimizer_{}) : output(output), stop(stop)
    {
    }

    virtual void init ()
    {
        output.initialize();
        stop.initialize();
    }

    virtual V clone () const { return std::unique_ptr<Optimizer>(new Optimizer(*this)); }

    virtual V optimize (::nlpp::wrap::poly::Function<V>, V) { return V{}; }


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


template <class V>
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

template <class V>
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

} // namespace nlpp::poly