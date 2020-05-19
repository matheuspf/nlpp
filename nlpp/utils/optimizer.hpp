/** @file
 *  @brief Optimizer base CRTP class
 * 
 *  @details Expose some default interface, wrapping function and gradient calls before delegating to
 *           the actual implementation. Also defines base parameter class for optimizers.
*/

#pragma once



//#include "../LineSearch/Goldstein/Goldstein.hpp"

#include "helpers/helpers.hpp"
#include "utils/finiteDifference.hpp"
#include "utils/wrappers.hpp"
#include "utils/output.hpp"
#include "utils/stop.hpp"

#include "../LineSearch/LineSearch.hpp"
#include "../LineSearch/StrongWolfe/StrongWolfe.hpp"


/// Base parameter definitions
#define NLPP_USING_OPTIMIZER(TYPE, ...) using TYPE = __VA_ARGS__;    \
                                        using TYPE::TYPE;            \
                                        using Stop = typename TYPE::Stop; \
                                        using Output = typename TYPE::Output; \
                                        using TYPE::stop;   \
                                        using TYPE::output;

#define NLPP_USING_GRADIENT_OPTIMIZER(TYPE, ...) NLPP_USING_OPTIMIZER(TYPE, __VA_ARGS__)

#define NLPP_USING_LINESEARCH_OPTIMIZER(TYPE, ...) NLPP_USING_GRADIENT_OPTIMIZER(TYPE, __VA_ARGS__) \
                                                   using LineSearch = typename TYPE::LineSearch;    \
                                                   using TYPE::lineSearch;

#define NLPP_USING_BOUND_CONSTRAINED_OPTIMIZER(TYPE, ...) NLPP_USING_OPTIMIZER(TYPE, __VA_ARGS__)

namespace nlpp
{

namespace impl
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

using ::nlpp::wrap::Functions, ::nlpp::wrap::Domain, ::nlpp::wrap::Constraints;

template <class Impl>
struct Optimizer
{
    using LineSearch = typename traits::Optimizer<Impl>::LineSearch;
    using Stop = typename traits::Optimizer<Impl>::Stop;
    using Output = typename traits::Optimizer<Impl>::Output;

    Optimizer (const LineSearch& lineSearch = LineSearch{}, const Stop& stop = Stop{}, const Output& output = Output{}) : lineSearch(lineSearch), stop(stop), output(output)
    {
    }

    template <class... Args>
    impl::Plain<V> operator () (const Args&... args) const
    {
        // return opt<V>(...);
    }

    // template <class V, class... FunctionArgs, class... DomainArgs, class... ConstraintArgs>
    // impl::Plain<V> opt (const Functions<V, FunctionArgs...>& functions, const Domain<V, DomainArgs...>& domain, const Constraints<V, ConstraintArgs...>& constraints) const
    // {
    //     return static_cast<const Impl&>(*this).optimize(functions, domain, constraints);
    // }

    template <class V, class Functions, class Domain, class Constraints>
    impl::Plain<V> opt (const Functions& functions, const Domain& domain, const Constraints& constraints) const
    {
        return static_cast<const Impl&>(*this).optimize(functions, domain, constraints);
    }


    LineSearch lineSearch;
    Stop stop;      ///< Stopping condition
    Output output;  ///< The output callback
};

} // namespace impl
//@}

template <class Impl>
using Optimizer = impl::Optimizer<Impl, wrap::Builder>;


namespace poly
{

using ::nlpp::impl::Scalar, ::nlpp::impl::Plain;

template <class V = ::nlpp::Vec>
struct Optimizer
{
    using LineSearch = ::nlpp::poly::LineSearch_<V>;
    using Stop = ::nlpp::poly::stop::GradientOptimizer_<V>;
    using Output = ::nlpp::poly::out::GradientOptimizer_<V>;

    using Functions = ::nlpp::wrap::poly::Functions<V>;
    using Domain = ::nlpp::wrap::poly::Domain<V>;
    using Constraints = ::nlpp::wrap::poly::Constraints<V>;

    Optimizer (const LineSearch& lineSearch = LineSearch{}, const Stop& stop = Stop{}, const Output& output = Output{}) : lineSearch(lineSearch), stop(stop), output(output)
    {
    }


    template <class... Args>
    impl::Plain<V> operator () (const Args&... args) const
    {
        // return optimize(...);
    }

    virtual impl::Plain<V> optimize (const Functions&, const Domain&, const Constraints&) const = 0;


    LineSearch lineSearch;
    Stop stop;      ///< Stopping condition
    Output output;  ///< The output callback
};

} // namespace poly


} // namespace nlpp