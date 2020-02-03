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

template <class Impl, template <class> class Builder>
struct Optimizer
{
    using Stop = typename traits::Optimizer<Impl>::Stop;
    using Output = typename traits::Optimizer<Impl>::Output;

    Optimizer (const Stop& stop = Stop{}, const Output& output = Output{}) : stop(stop), output(output)
    {
    }

    void initialize ()
    {
        stop.initialize();
        output.initialize();
    }

    template <class Function, class V, typename... Args>
    impl::Plain<V> operator () (const Function& function, const Eigen::MatrixBase<V>& x, Args&&... args) const
    {
        return static_cast<const Impl&>(*this).optimize(Builder<V>::function(function), x.eval(), std::forward<Args>(args)...);
    }

    Stop stop;      ///< Stopping condition
    Output output;  ///< The output callback
};

template <class Impl, template <class> class Builder>
struct GradientOptimizer : public Optimizer<Impl, Builder>
{
    NLPP_USING_OPTIMIZER(Base, Optimizer<Impl, Builder>);

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
        return static_cast<Impl&>(*this).optimize(Builder<V>::functionGradient(function, gradient), x.eval(), std::forward<Args>(args)...);
    }

    template <class Function, class V, typename... Args>
    impl::Plain<V> operator () (const Function& function, const Eigen::MatrixBase<V>& x, Args&&... args)
    {
        return static_cast<Impl&>(*this).optimize(Builder<V>::functionGradient(function), x.eval(), std::forward<Args>(args)...);
    }
};

template <class Impl, template <class> class Builder>
struct HessianOptimizer : public GradientOptimizer<Impl, Builder>
{
    NLPP_USING_GRADIENT_OPTIMIZER(Base, GradientOptimizer<Impl, Builder>);

    template <class Function, class Gradient, class V, typename... Args>
    impl::Plain<V> operator () (const Function& function, const Gradient& gradient, const Eigen::MatrixBase<V>& x, Args&&... args)
    {
        return static_cast<Impl&>(*this).optimize(Builder<V>::functionGradient(function, gradient), Builder<V>::hessian(function), x.eval(), std::forward<Args>(args)...);
    }

    template <class Function, class V, typename... Args>
    impl::Plain<V> operator () (const Function& function, const Eigen::MatrixBase<V>& x, Args&&... args)
    {
        return static_cast<Impl&>(*this).optimize(Builder<V>::functionGradient(function), Builder<V>::hessian(function), x.eval(), std::forward<Args>(args)...);
    }
};

template <class Impl, template <class> class Builder>
struct BoundConstrainedOptimizer : public Optimizer<Impl, Builder>
{
    NLPP_USING_OPTIMIZER(Base, Optimizer<Impl, Builder>);

    template <class Function, class V, typename... Args>
    impl::Plain<V> operator () (const Function& function, const Eigen::MatrixBase<V>& lower,  const Eigen::MatrixBase<V>& upper, Args&&... args)
    {
        return static_cast<Impl&>(*this).optimize(Builder<V>::function(function), lower.eval(), upper.eval(), std::forward<Args>(args)...);
    }
};

template <template <class> class BaseOptimizer, class Impl>
struct LineSearchOptimizer : public BaseOptimizer<Impl>
{
    NLPP_USING_GRADIENT_OPTIMIZER(Base, BaseOptimizer<Impl>);
    using LineSearch = typename traits::Optimizer<Impl>::LineSearch;

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

} // namespace impl
//@}

template <class Impl>
using Optimizer = impl::Optimizer<Impl, wrap::Builder>;

template <class Impl>
using GradientOptimizer = impl::GradientOptimizer<Impl, wrap::Builder>;

template <class Impl>
using HessianOptimizer = impl::HessianOptimizer<Impl, wrap::Builder>;

template <class Impl>
using BoundConstrainedOptimizer = impl::BoundConstrainedOptimizer<Impl, wrap::Builder>;

template <class Impl>
using LineSearchOptimizer = impl::LineSearchOptimizer<GradientOptimizer, Impl>;


namespace poly
{

using ::nlpp::impl::Scalar, ::nlpp::impl::Plain;

template <class V = ::nlpp::Vec>
struct Optimizer : ::nlpp::impl::Optimizer<Optimizer<V>, ::nlpp::wrap::poly::Builder>
{
    NLPP_USING_OPTIMIZER(Base, ::nlpp::impl::Optimizer<Optimizer<V>, ::nlpp::wrap::poly::Builder>);

    virtual void initialize ()
    {
        Base::initialize();
    }

    virtual V optimize (const ::nlpp::wrap::poly::Function<V>&, const V&) = 0;
};

template <class V = ::nlpp::Vec>
struct GradientOptimizer : ::nlpp::impl::GradientOptimizer<GradientOptimizer<V>, ::nlpp::wrap::poly::Builder>
{
    NLPP_USING_OPTIMIZER(Base, ::nlpp::impl::GradientOptimizer<GradientOptimizer<V>, ::nlpp::wrap::poly::Builder>);

    virtual void initialize ()
    {
        Base::initialize();
    }

    virtual V optimize (const ::nlpp::wrap::poly::FunctionGradient<V>&, const V&) = 0;
};

template <class V = ::nlpp::Vec>
struct BoundConstrainedOptimizer : ::nlpp::impl::BoundConstrainedOptimizer<BoundConstrainedOptimizer<V>, ::nlpp::wrap::poly::Builder>
{
    NLPP_USING_BOUND_CONSTRAINED_OPTIMIZER(Base, ::nlpp::impl::BoundConstrainedOptimizer<BoundConstrainedOptimizer<V>, ::nlpp::wrap::poly::Builder>);

    virtual void initialize ()
    {
        Base::initialize();
    }

    virtual V optimize (const ::nlpp::wrap::poly::Function<V>&, const V&, const V&) = 0;
};

template <class V = ::nlpp::Vec>
struct LineSearchOptimizer : GradientOptimizer<V>
{
    NLPP_USING_GRADIENT_OPTIMIZER(Base, GradientOptimizer<V>);
    using LineSearch = ::nlpp::poly::LineSearch_<V>;

    virtual void initialize ()
    {
        Base::initialize();
        lineSearch.initialize();
    }

    LineSearch lineSearch;
};

} // namespace poly

namespace traits
{

template <class V>
struct Optimizer<::nlpp::poly::Optimizer<V>>
{
    using Stop = ::nlpp::poly::stop::Optimizer_<V>;
    using Output = ::nlpp::poly::out::Optimizer_<V>;
};

template <class V>
struct Optimizer<::nlpp::poly::GradientOptimizer<V>>
{
    using Stop = ::nlpp::poly::stop::GradientOptimizer_<V>;
    using Output = ::nlpp::poly::out::GradientOptimizer_<V>;
};

template <class V>
struct Optimizer<::nlpp::poly::LineSearchOptimizer<V>>
{
    using LineSearch = ::nlpp::poly::LineSearch_<V>;
    using Stop = ::nlpp::poly::stop::GradientOptimizer_<V>;
    using Output = ::nlpp::poly::out::GradientOptimizer_<V>;
};

template <class V>
struct Optimizer<::nlpp::poly::BoundConstrainedOptimizer<V>>
{
    using Stop = ::nlpp::poly::stop::Optimizer_<V>;
    using Output = ::nlpp::poly::out::Optimizer_<V>;
};

} // namespace traits
 
} // namespace nlpp