#pragma once

#include "Types.h"


namespace nlpp
{

/** @brief Some forward declarations
*/
//@{
namespace out
{

template <int Level = 0, typename Float = types::Float>
struct Optimizer;

template <int Level = 0, typename Float = types::Float>
struct GradientOptimizer;

namespace poly
{

template <class V = ::nlpp::Vec>
struct Optimizer_;

template <class V = ::nlpp::Vec>
struct GradientOptimizer_;

} // namespace poly

} // namespace out


namespace stop
{

template <bool Exclusive = false, typename Float = types::Float>
struct Optimizer;

template <bool Exclusive = false, typename Float = types::Float>
struct GradientOptimizer;

namespace poly
{

template <class V = ::nlpp::Vec>
struct Optimizer_;

template <class V = ::nlpp::Vec>
struct GradientOptimizer_;

} // namespace poly

} // namespace stop



/// Because these are used as the default line search procedure in many cases
template <typename Float = types::Float>
struct ConstantStep;

template <typename Float = types::Float, class InitialStep = ConstantStep<Float>>
struct StrongWolfe;

namespace poly
{

template <typename Float = types::Float>
struct LineSearch;

template <typename Float = types::Float>
struct LineSearch_;

template <typename Float = types::Float, class InitialStep = ConstantStep<Float>>
struct StrongWolfe;

} // namespace poly


template <class = stop::Optimizer<>, class = out::Optimizer<>>
struct Optimizer;

template <class = stop::GradientOptimizer<>, class = out::GradientOptimizer<>>
struct GradientOptimizer;

template <class = StrongWolfe<>, class = stop::GradientOptimizer<>, class = out::GradientOptimizer<>>
struct LineSearchOptimizer;

namespace poly
{

template <class V = Vec>
struct Optimizer;

template <class V = Vec>
struct GradientOptimizer;

template <class V = Vec>
struct LineSearchOptimizer;

} // namespace poly

//@}


namespace fd
{

struct AutoStep;

template <typename Float = types::Float>
struct SimpleStep;

template <class Function, class Step = AutoStep>
struct Forward;

template <class Function, template <class, class> class Difference = Forward, class Step = AutoStep>
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


template <class T, class V = Vec>
struct FunctionType;

template <class T, class V = Vec>
struct GradientType;

template <class T, class V = Vec>
struct FunctionGradientType;

template <class T, class V = Vec, class U = Vec>
struct HessianType;


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
        std::conditional_t<wrap::FunctionType<Func>::value >= 0 && wrap::FunctionGradientType<Func>::value < 0,
            impl::FunctionGradient<Func, fd::Gradient<Func>>,
            impl::FunctionGradient<Func>
        >
    >,
    impl::FunctionGradient<Func, Grad>
>;


} // namespace wrap


} // namespace nlpp