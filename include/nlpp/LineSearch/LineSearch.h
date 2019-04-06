/** @file
 * 	@brief Line search base class for CRTP
*/

#pragma once

#include "../Helpers/Helpers.h"

#include "../Helpers/Wrappers.h"

#include "../Helpers/FiniteDifference.h"

#include "InitialStep/Constant.h"



namespace nlpp
{

namespace wrap
{

/// A utility to wrap a vector function to a scalar function along a direction @c d
template <class FunctionGradient, class V>
struct LineSearch
{
	using Float = typename V::Scalar;

	LineSearch (const FunctionGradient& f, const V& x, const V& d) : f(f), x(x), d(d), gx(x.rows(), x.cols())
	{
	}

	std::pair<Float, Float> operator () (Float a)
	{
		// auto fx = f(x + a * d, gx);

		// return std::make_pair(fx, gx.dot(d));

		V xn = x + a * d;

		Float fx = f.function(xn);

		Float gx = f.directional(xn, d, fx);

		return std::make_pair(fx, gx);
	}

	Float function (Float a)
	{
		return f.function(x + a * d);
	}

	Float gradient (Float a)
	{
		// f.gradient(x + a * d, gx);

		// return gx.dot(d);

		return f.directional(x, d);
	}


	FunctionGradient f;

	const V& x;
	const V& d;

	typename V::PlainObject gx;
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
    void initialize () 
	{
		static_cast<Impl&>(*this).initialize();
	}

    template <class Function, class V>
	auto impl (const Function& f, const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& dir)
	{
		return static_cast<Impl&>(*this).lineSearch(wrap::LineSearch<Function, V>(f, x, dir));
	}

	template <class Function, class V>
	auto operator () (const Function& f, const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& dir)
	{
	    return impl(wrap::functionGradient(f), x, dir);
	}

	template <class Function, class Gradient, class V>
	auto operator () (const Function& f, Gradient g, const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& dir)
	{
		return impl(wrap::functionGradient(f, g), x, dir);
	}
};



namespace poly
{

template <class Impl, typename Float = types::Float>
struct LineSearchCRTP
{
	using V = ::nlpp::VecX<Float>;

   	void initialize () 
	{
		static_cast<Impl&>(*this).initialize();
	}

	template <class Function>
	Float impl (const Function& f, const V& x, const V& dir)
	{
		return static_cast<Impl&>(*this).lineSearch(wrap::LineSearch<Function, V>(f, x, dir));
	}

	template <class Function>
	Float operator () (const Function& f, const V& x, const V& dir)
	{
		return impl(wrap::poly::FunctionGradient<V>(f), x, dir);
	}

	template <class Function, class Gradient>
	Float operator () (const Function& f, const Gradient& g, const V& x, const V& dir)
	{
		return impl(wrap::poly::FunctionGradient<V>(f, g), x, dir);
    }
};


template <typename Float>
struct LineSearch : public ::nlpp::poly::CloneBase<LineSearch<Float>>,
					public LineSearchCRTP<LineSearch<Float>, Float>
{
	virtual ~LineSearch ()	{}

    using V = ::nlpp::VecX<Float>;


	virtual void initialize () = 0;

	virtual Float lineSearch (::nlpp::wrap::LineSearch<::nlpp::wrap::poly::FunctionGradient<V>, V>) = 0;
};


template <typename Float>
struct LineSearch_ : public ::nlpp::poly::PolyClass<LineSearch<Float>>,
					 public LineSearchCRTP<LineSearch_<Float>, Float>
{
	NLPP_USING_POLY_CLASS(LineSearch_, Base, ::nlpp::poly::PolyClass<LineSearch<Float>>);

    using V = typename LineSearch<Float>::V;


	LineSearch_ () : Base(std::make_unique<StrongWolfe<Float>>()) {}

    void initialize ()
	{
		return impl->initialize();
	}

	Float lineSearch (::nlpp::wrap::LineSearch<::nlpp::wrap::poly::FunctionGradient<V>, V> f)
	{
		return impl->lineSearch(f);
	}
};


} // namespace poly



template <typename Float = types::Float, class InitialStep = ConstantStep<Float>>
struct LineSearchBase
{
	LineSearchBase (const InitialStep& initialStep) : initialStep(initialStep) {}


    void initialize ()
	{
		initialStep.initialize();
	}


	Float aStart;
	Float f0;
	Float g0;

	InitialStep initialStep;
};



} // namespace nlpp
