/** @file
 * 	@brief Line search base class for CRTP
*/

#pragma once

#include "helpers/helpers.hpp"
#include "utils/wrappers/functions.hpp"
#include "utils/finite_difference_dec.hpp"
#include "initial_step/constant.hpp"

#define NLPP_USING_LINESEARCH(TYPE, ...)            \
    using TYPE = __VA_ARGS__;                       \
    using TYPE::TYPE;                               \
    using Float = typename TYPE::Float;             \
    using InitialStep = typename TYPE::InitialStep; \
    using TYPE::initialStep;


namespace nlpp::ls
{

namespace wrap
{

/// A utility to wrap a vector function to a scalar function along a direction @c d
template <class Functions, class V, class U>
struct LineSearch
{
    using Float = ::nlpp::impl::Scalar<V>;

    LineSearch(const Functions &f, const V &x, const U &d) : f(f), x(x), d(d)
    {
    }

    std::pair<Float, Float> operator()(Float a) const
    {
        return funcGrad(a);
    }

    std::pair<Float, Float> funcGrad (Float a) const
    {
        auto xn = x + a * d;

        Float fx = f.function(xn);
        auto gx = f.gradientDir(xn, d, fx);

        return std::make_pair(fx, gx);
    }

    Float function(Float a) const
    {
        return f.function(x + a * d);
    }

    Float gradient(Float a) const
    {
        return f.gradientDir(x, d);
    }

    Float gradient(Float a, Float fx) const
    {
        return f.gradientDir(x, d, fx);
    }

    const Functions &f;
    const V &x;
    const U &d;
};

} // namespace wrap

/** @brief Line search base class for CRTP
 *  
 *  @details Delegate the call to the base class @c Impl after projecting the given function and gradient into
 * 			 the direction dir. That is:
 * 
 * 			 - @f$ f'(a) = f(x + a * dir) @f$
 * 			 - @f$ g'(a) = g(x + a * dir) \intercall dir @f$
 * 
 * 			So we can now use f' and g' exactly as if they were unidimensional scalar functions. Also, wraps the gradient 
 * 			or function/gradient calls before projection.
 * 
 *  @tparam Impl The actual line search implementation
 * 	@tparam Whether we must save the norm of the given vector before delegating the calls
*/
template <class Impl>
struct LineSearch
{
    using Float = typename traits::LineSearch<Impl>::Float;
    using InitialStep = typename traits::LineSearch<Impl>::InitialStep;

    LineSearch(const InitialStep &initialStep = InitialStep{}) : initialStep(initialStep) {}

    template <class Functions, class V, class U>
    ::nlpp::impl::Scalar<V> operator()(const Functions &functions, const Eigen::MatrixBase<V> &x, const Eigen::MatrixBase<U> &dir)
    {
        return static_cast<Impl&>(*this).lineSearch(wrap::LineSearch<Functions, V, U>(functions, x, dir));
    }

    InitialStep initialStep;
};

// namespace poly
// {

// template <class V = ::nlpp::Vec>
// struct LineSearchBase : public ::nlpp::poly::CloneBase<LineSearchBase<V>>
// {
// 	virtual ~LineSearchBase ()	{}

// 	virtual void initialize () = 0;

// 	virtual ::nlpp::impl::Scalar<V> lineSearch (::nlpp::wrap::LineSearch<::nlpp::wrap::poly::FunctionGradient<V>, V>) = 0;
// };

// template <class V = ::nlpp::Vec>
// struct LineSearch_ : public ::nlpp::poly::PolyClass<LineSearchBase<V>>,
// 					 public ::nlpp::impl::LineSearch<LineSearch_<V>, ::nlpp::wrap::poly::Builder>
// {
// 	NLPP_USING_POLY_CLASS(LineSearch_, Base, ::nlpp::poly::PolyClass<LineSearchBase<V>>);

// 	LineSearch_ () : Base(std::make_unique<StrongWolfe<V, ConstantStep<::nlpp::impl::Scalar<V>>>>()) {}

//     void initialize ()
// 	{
// 		return impl->initialize();
// 	}

// 	::nlpp::impl::Scalar<V> lineSearch (::nlpp::wrap::LineSearch<::nlpp::wrap::poly::FunctionGradient<V>, V> f)
// 	{
// 		return impl->lineSearch(f);
// 	}
// };

// } // namespace poly

} // namespace nlpp::ls
