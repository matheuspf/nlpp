#pragma once

#include "Types.h"


namespace nlpp
{

/** @brief Some forward declarations
*/
//@{
namespace out
{

template <int = 0, typename = types::Float>
struct GradientOptimizer;

} // namespace out


namespace stop
{

template <bool = false, typename = types::Float>
struct GradientOptimizer;

} // namespace stop


/// Because this is used as the default line search procedure in many cases
struct StrongWolfe;


namespace params
{

template <class = StrongWolfe, class = stop::GradientOptimizer<>, class = out::GradientOptimizer<>>
struct GradientOptimizer;

} // namespace params
//@}


namespace fd
{

template <typename Float = types::Float>
struct SimpleStep;

template <class Function, class Step = SimpleStep<>>
struct Forward;

template <class Function, template <class, class> class Difference = Forward, class Step = SimpleStep<>>
struct Gradient;

} // namespace fd


namespace wrap
{

namespace impl
{

template <class Impl>
struct Function;

template <class Impl>
struct Gradient;

template <class...>
struct FunctionGradient;

} // namespace impl


template <class T, class V = Mat>
struct IsFunction;

template <class T, class V = Mat>
struct IsGradient;

template <class T, class V = Mat>
struct IsFunctionGradient;



template <class Impl>
using Function = std::conditional_t<handy::IsSpecialization<Impl, impl::Function>::value, Impl, impl::Function<Impl>>;

template <class Impl>
using Gradient = std::conditional_t<handy::IsSpecialization<Impl, impl::Gradient>::value, Impl, impl::Gradient<Impl>>;

/** @brief Alias for impl::FunctionGradient
 *  @details There are four conditions:
 *           - Is @c Grad given?
 *              - If not, is @c Func already an impl::FunctionGradient?
 *                  - (1) If so, simply set the result to itself (to avoid multiple wrapping)
 *                  - Otherwise, is @c Func a function but not a function/gradient functor?
 *                      - (2) If so, set the result to impl::FunctionGradient<Func, fd::Gradient<Func>> (use finite difference to aproximate the gradient)
 *                      - (3) Otherwise it should be a function/gradient functor, so we use impl::FunctionGradient<Func>
 *              - (4) If yes, set the result to impl::FunctionGradient<Func, Grad>
*/
template <class Func, class Grad = void>
using FunctionGradient = std::conditional_t<std::is_same<Grad, void>::value,
    std::conditional_t<handy::IsSpecialization<Func, impl::FunctionGradient>::value,
        Func,
        std::conditional_t<wrap::IsFunction<Func>::value >= 0 && wrap::IsFunctionGradient<Func>::value < 0,
            impl::FunctionGradient<Func, fd::Gradient<Func>>,
            impl::FunctionGradient<Func>
        >
    >,
    impl::FunctionGradient<Func, Grad>
>;


} // namespace wrap


} // namespace nlpp