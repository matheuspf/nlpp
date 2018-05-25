#pragma once


#include "Helpers.h"

#include "FiniteDifference.h"

#include "Gradient.h"

#include "../LineSearch/StrongWolfe/StrongWolfe.h"


#define CPPOPT_USING_PARAMS(TYPE, ...) using TYPE = __VA_ARGS__;    \
                                       using TYPE::maxIterations;   \
                                       using TYPE::xTol;            \
                                       using TYPE::fTol;            \
                                       using TYPE::gTol;            \
                                       using TYPE::lineSearch;




namespace cppnlp
{


namespace params
{


template <class LineSearch = StrongWolfe>
struct GradientOptimizer
{
    GradientOptimizer(int maxIterations = 1000, double xTol = 1e-4, double fTol = 1e-4, 
                      double gTol = 1e-4, const LineSearch& lineSearch = LineSearch()) :
                      maxIterations(maxIterations), xTol(xTol), fTol(fTol), gTol(gTol), lineSearch(lineSearch) {}

    GradientOptimizer(const LineSearch& lineSearch, int maxIterations = 1000, double xTol = 1e-4, double fTol = 1e-4, double gTol = 1e-4) :
                    maxIterations(maxIterations), xTol(xTol), fTol(fTol), gTol(gTol), lineSearch(lineSearch) {}

    int maxIterations;

	double xTol;

	double fTol;

    double gTol;


    LineSearch lineSearch;
};


} // namespace params



template <class Impl, class Parameters = params::GradientOptimizer<>>
struct GradientOptimizer : public Parameters
{
    GradientOptimizer(const Parameters& params = Parameters()) : Parameters(params) {}


    template <class T, std::enable_if_t<!std::is_fundamental<std::result_of_t<T(const Vec&)>>::value, int> = 0>
    static constexpr int testRet (T);

    template <class T, std::enable_if_t<std::is_fundamental<std::result_of_t<T(const Vec&)>>::value, int> = 0>
    static constexpr void testRet (T);

    static constexpr std::nullptr_t testRet (...);

    template <class T>
    static constexpr bool TestRet = std::is_same<decltype(testRet(std::declval<T>())), int>::value;
    

    
    template <class Function, class Gradient, class Vec, typename... Args>
    Vec operator () (const Function& function, const Gradient& gradient, const Eigen::MatrixBase<Vec>& x, Args&&... args)
    {
        return static_cast<Impl&>(*this).optimize(wrap::functionGradient(function, gradient), x.eval(), std::forward<Args>(args)...);
    }

    template <class FunctionGradient, class Vec, typename... Args,
              std::enable_if_t<wrap::HasOp<FunctionGradient, const Vec&, Vec&>::value || 
                               TestRet<FunctionGradient>, int> = 0>
    Vec operator () (const FunctionGradient& funcGrad, const Eigen::MatrixBase<Vec>& x, Args&&... args)
    {
        return static_cast<Impl&>(*this).optimize(wrap::functionGradient(funcGrad), x.eval(), std::forward<Args>(args)...);
    }
    

    template <class Function, class Vec, typename... Args, 
              std::enable_if_t<!TestRet<Function> && !wrap::HasOp<Function, const Vec&, Vec&>::value, int> = 0>
    Vec operator () (const Function& func, const Eigen::MatrixBase<Vec>& x, Args&&... args)
    {
        return operator()(func, fd::gradient(func), x, std::forward<Args>(args)...);
    }
};




} // namespace cppnlp