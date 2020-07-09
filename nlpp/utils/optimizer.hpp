/** @file
 *  @brief Optimizer base CRTP class
 * 
 *  @details Expose some default interface, wrapping function and gradient calls before delegating to
 *           the actual implementation. Also defines base parameter class for optimizers.
*/

#pragma once



//#include "../LineSearch/Goldstein/Goldstein.hpp"

#include "helpers/helpers.hpp"
#include "utils/finite_difference_dec.hpp"
#include "utils/wrappers/functions.hpp"
#include "utils/wrappers/domain.hpp"
#include "utils/wrappers/constraints.hpp"
#include "utils/output.hpp"
#include "utils/stop.hpp"


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


template <std::size_t...> class TTT;

using ::nlpp::wrap::Functions, ::nlpp::wrap::Domain, ::nlpp::wrap::Constraints;

template <class Impl>
struct LineSearchOptimizer
{
    using LineSearch = typename traits::Optimizer<Impl>::LineSearch;
    using Stop = typename traits::Optimizer<Impl>::Stop;
    using Output = typename traits::Optimizer<Impl>::Output;

    static constexpr wrap::Conditions conditions = traits::Optimizer<Impl>::conditions;


    template <class... Fs>
    static constexpr auto functions (Fs&&... fs)
    {
        return wrap::functions<conditions>(std::forward<Fs>(fs)...);
    }

    template <class... Vs>
    static constexpr auto domain (Vs&&... vs)
    {
        return wrap::domain<conditions>(std::forward<Vs>(vs)...);
    }

    template <class... Fs>
    static constexpr auto constraints (Fs&&... fs)
    {
        return wrap::constraints<conditions>(std::forward<Fs>(fs)...);
    }



    LineSearchOptimizer (const LineSearch& lineSearch = LineSearch{}, const Stop& stop = Stop{}, const Output& output = Output{}) : lineSearch(lineSearch), stop(stop), output(output)
    {
    }



    template <class... Args>
    auto operator () (Args&&... args) const
    {
        return impl(std::make_index_sequence<sizeof...(Args)>{}, std::forward<Args>(args)...);
    }

    template <class Functions, class Domain, class Constraints>
    auto opt (const Functions& functions, const Domain& domain, const Constraints& constraints) const
    {
        return static_cast<const Impl&>(*this).optimize(functions, domain, constraints);
    }


    template <std::size_t... IArgs, class... Args>
    constexpr auto impl (std::index_sequence<IArgs...>, Args&&... args) const
    {
        constexpr std::size_t StartDomain = std::min({ ((IArgs + 1) * (::nlpp::impl::isEigen<Args> ? 1 : 100))... }) - 1;
        constexpr std::size_t EndDomain = std::max({ ((IArgs + 1) * ::nlpp::impl::isEigen<Args>)... }) - 1;

        return impl2(std::make_index_sequence<StartDomain>{},
                     std::make_index_sequence<EndDomain - StartDomain + 1>{},
                     std::make_index_sequence<sizeof...(Args) - EndDomain - 1>{},
                     std::forward<Args>(args)...);
    }

    template <std::size_t... IFunctions, std::size_t... IDomain, std::size_t... IConstraints, class... Args>
    constexpr auto impl2 (std::index_sequence<IFunctions...>, std::index_sequence<IDomain...>, std::index_sequence<IConstraints...>, Args&&... args) const
    {
        auto tup = std::forward_as_tuple(std::forward<Args>(args)...);

        return opt(functions(std::get<IFunctions>(tup)...), 
                   domain(std::get<sizeof...(IFunctions) + IDomain>(tup)...),
                   constraints(std::get<sizeof...(IFunctions) + sizeof...(IDomain) + IConstraints>(tup)...));
    }


    LineSearch lineSearch;
    Stop stop;      ///< Stopping condition
    Output output;  ///< The output callback
};

} // namespace impl
//@}


template <class Impl>
using LineSearchOptimizer = impl::LineSearchOptimizer<Impl>;



// namespace poly
// {

// using ::nlpp::impl::Scalar, ::nlpp::impl::Plain;

// template <class V = ::nlpp::Vec>
// struct Optimizer
// {
//     using LineSearch = ::nlpp::poly::LineSearch_<V>;
//     using Stop = ::nlpp::poly::stop::GradientOptimizer_<V>;
//     using Output = ::nlpp::poly::out::GradientOptimizer_<V>;

//     using Functions = ::nlpp::wrap::poly::Functions<V>;
//     using Domain = ::nlpp::wrap::poly::Domain<V>;
//     using Constraints = ::nlpp::wrap::poly::Constraints<V>;

//     Optimizer (const LineSearch& lineSearch = LineSearch{}, const Stop& stop = Stop{}, const Output& output = Output{}) : lineSearch(lineSearch), stop(stop), output(output)
//     {
//     }


//     template <class... Args>
//     impl::Plain<V> operator () (const Args&... args) const
//     {
//         // return optimize(...);
//     }

//     virtual impl::Plain<V> optimize (const Functions&, const Domain&, const Constraints&) const = 0;


//     LineSearch lineSearch;
//     Stop stop;      ///< Stopping condition
//     Output output;  ///< The output callback
// };

// } // namespace poly


} // namespace nlpp