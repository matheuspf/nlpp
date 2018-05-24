#pragma once


#include "Helpers.h"

#include "FiniteDifference.h"

#include "Gradient.h"


namespace cppnlp
{

template <class Impl>
struct GradientOptimizer
{
    template <class Function, class Gradient, class Vec, typename... Args>
    Vec operator () (const Function& function, const Gradient& gradient, const Eigen::MatrixBase<Vec>& x, Args&&... args)
    {
        return static_cast<Impl&>(*this).optimize(wrap::functionGradient(function, gradient), x.eval(), std::forward<Args>(args)...);
    }

    template <class FunctionGradient, class Vec, typename... Args,
              std::enable_if_t<1 || 
              (wrap::HasOverload<FunctionGradient, Vec>::value && 
               !std::is_fundamental<std::result_of_t<FunctionGradient(Vec)>>::value), int> = 0>
    Vec operator () (const FunctionGradient& funcGrad, const Eigen::MatrixBase<Vec>& x, Args&&... args)
    {
        return static_cast<Impl&>(*this).optimize(wrap::functionGradient(funcGrad), x.eval(), std::forward<Args>(args)...);
    }

    template <class Function, class Vec, typename... Args,
              std::enable_if_t<wrap::HasOverload<Function, Vec>::value && 
              std::is_fundamental<std::result_of_t<Function(Vec)>>::value, int> = 0>
    Vec operator () (const Function& func, const Eigen::MatrixBase<Vec>& x, Args&&... args)
    {
        return operator()(func, fd::gradient(func), x, std::forward<Args>(args)...);
    }
};

} // namespace cppnlp