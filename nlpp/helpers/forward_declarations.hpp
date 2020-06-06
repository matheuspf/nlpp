#pragma once

#include "types.hpp"


namespace nlpp
{

/** @brief Some forward declarations
*/
//@{
namespace out
{

template <int Level, typename Float>
struct Optimizer;

template <int Level, typename Float>
struct GradientOptimizer;

namespace poly
{

template <class V>
struct Optimizer_;

template <class V>
struct GradientOptimizer_;

} // namespace poly
} // namespace out


namespace stop
{

template <bool Exclusive, typename Float>
struct Optimizer;

template <bool Exclusive, typename Float>
struct GradientOptimizer;

namespace poly
{

template <class V>
struct Optimizer_;

template <class V>
struct GradientOptimizer_;

} // namespace poly
} // namespace stop



/// Because these are used as the default line search procedure in many cases
template <typename Float>
struct ConstantStep;

template <typename Float, class InitialStep>
struct StrongWolfe;

namespace poly
{

template <class V>
struct LineSearchBase;

template <class V>
struct LineSearch_;

template <class V, class InitialStep>
struct StrongWolfe;

} // namespace poly


namespace poly
{

template <class V>
struct Optimizer;

template <class V>
struct GradientOptimizer;

template <class V>
struct LineSearchOptimizer;

} // namespace poly

//@}


namespace fd
{

struct AutoStep;

template <typename Float>
struct SimpleStep;

template <typename Float>
struct NormalizedStep;

template <class Function, class Step>
struct Forward;

template <class Function, class Step>
struct Backward;

template <class Function, class Step>
struct Central;

template <class Function, template <class, class> class Difference, class Step>
struct Gradient;

template <class Function, template <class, class> class Difference, class Step, typename Float>
struct Hessian;

} // namespace fd


namespace wrap::impl
{

template <class Impl>
struct Function;

template <class Impl>
struct Gradient;

template <class Impl>
struct FunctionGradient;

template <class>
struct Hessian;

} // namespace wrap::impl


namespace traits
{

template <class>
struct Optimizer;

template <class>
struct LineSearch;

} // namespace traits

} // namespace nlpp