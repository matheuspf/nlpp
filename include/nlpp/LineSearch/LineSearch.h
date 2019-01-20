/** @file
 * 	@brief Line search base class for CRTP
*/

#pragma once

#include "../Helpers/Helpers.h"

#include "../Helpers/Wrappers.h"

#include "../Helpers/FiniteDifference.h"

#include "../Helpers/Optimizer.h"


#define NLPP_TEMPLATE_PARAMS7(T7, ...) typename T7, NLPP_TEMPLATE_PARAMS6(__VA_ARGS__)
#define NLPP_TEMPLATE_PARAMS6(T6, ...) typename T6, NLPP_TEMPLATE_PARAMS5(__VA_ARGS__) 
#define NLPP_TEMPLATE_PARAMS5(T5, ...) typename T5, NLPP_TEMPLATE_PARAMS4(__VA_ARGS__) 
#define NLPP_TEMPLATE_PARAMS4(T4, ...) typename T4, NLPP_TEMPLATE_PARAMS3(__VA_ARGS__) 
#define NLPP_TEMPLATE_PARAMS3(T3, ...) typename T3, NLPP_TEMPLATE_PARAMS2(__VA_ARGS__) 
#define NLPP_TEMPLATE_PARAMS2(T2, ...) typename T2, NLPP_TEMPLATE_PARAMS1(__VA_ARGS__) 
#define NLPP_TEMPLATE_PARAMS1(T1)      typename T1

#define NLPP_TEMPLATE_PARAMS(...) APPLY_N(NLPP_TEMPLATE_PARAMS, __VA_ARGS__)


#define NLPP_LINE_SEARCH_ALIASES(Name, ...)		using Interface = LineSearch<Name>;	\
												using Impl = impl::__VA_ARGS__;				\
												using Impl::Impl;						\


#define NLPP_LINE_SEARCH(Name, ...) NLPP_LINE_SEARCH_IMPL(Name, Float = types::Float, ## __VA_ARGS__)

#define NLPP_LINE_SEARCH_D(Name, ...) NLPP_LINE_SEARCH_IMPL(Name, Float, ## __VA_ARGS__)


#define NLPP_LINE_SEARCH_IMPL(Name, Float_Template, ...)	\
\
template <EXPAND(NLPP_TEMPLATE_PARAMS(Float_Template, ## __VA_ARGS__))>					\
struct Name : public impl::Name<Float, ## __VA_ARGS__>,					\
			  public LineSearch<Name<Float, ## __VA_ARGS__>>			\
{																		\
	NLPP_LINE_SEARCH_ALIASES(Name, Name<Float, ## __VA_ARGS__>)			\
																		\
	template <class Function>											\
	auto lineSearch (Function f)										\
	{																	\
		return Impl::lineSearch(f);										\
	}																	\
};																		\
																		\
namespace poly	\
{\
\
template <EXPAND(NLPP_TEMPLATE_PARAMS(Float_Template, ## __VA_ARGS__))> \
struct Name : public impl::Name<Float, ## __VA_ARGS__>, \
			  public LineSearch<Float>	\
{	\
	NLPP_LINE_SEARCH_ALIASES(Name, Name<Float, ## __VA_ARGS__>)	\
	\
	Float lineSearch (::nlpp::wrap::LineSearch<::nlpp::wrap::poly::FunctionGradient<>, ::nlpp::Vec> f)	\
	{	\
		return Impl::lineSearch(f);	\
	}	\
	\
	virtual Name* clone_impl () const {	return new Name(*this);	}	\
};	\
\
} // namespace poly



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
		auto fx = f(x + a * d, gx);

		return std::make_pair(fx, gx.dot(d));
	}

	Float function (Float a)
	{
		return f.function(x + a * d);
	}

	Float gradient (Float a)
	{
		f.gradient(x + a * d, gx);

		return gx.dot(d);
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
template <class Impl, bool CalcNorm = true>
struct LineSearch
{
	template <class Function, class Vec>
	double impl (Function f, const Eigen::MatrixBase<Vec>& x, const Eigen::MatrixBase<Vec>& dir)
	{
		return static_cast<Impl&>(*this).lineSearch(wrap::LineSearch<Function, Vec>(f, x, dir));
	}


	template <class FunctionGradient, class Vec, std::enable_if_t<(wrap::IsFunctionGradient<FunctionGradient, Vec>::value >= 0), int> = 0>
	double operator () (const FunctionGradient& f, const Eigen::MatrixBase<Vec>& x, const Eigen::MatrixBase<Vec>& dir)
	{
		return impl(f, x, dir);
	}

	template <class Function, class Gradient, class Vec>
	double operator () (Function f, Gradient g, const Eigen::MatrixBase<Vec>& x, const Eigen::MatrixBase<Vec>& dir)
	{
		return impl(wrap::functionGradient(f, g), x, dir);
	}

	template <class Function, class Vec, std::enable_if_t<(wrap::IsFunction<Function, Vec>::value >= 0 && wrap::IsFunctionGradient<Function, Vec>::value < 0), int> = 0>
	double operator () (Function f, const Eigen::MatrixBase<Vec>& x, const Eigen::MatrixBase<Vec>& dir)
	{
		return operator()(f, fd::gradient(f), x, dir);
	}


	template <class Function, class Gradient>
	double operator () (Function f, Gradient g, double x, double dir = 1.0)
	{
		return operator()([&](double a){ return f(x + a * dir); },
						  [&](double a){ return g(x + a * dir) * dir; });
	}

	template <class Function>
	double operator () (Function f, double x, double dir = 1.0)
	{
		return operator()(f, fd::gradient(f), x, dir);
	}


//private:

	LineSearch () {}

	friend Impl;
};



namespace poly
{

template <typename Float = ::nlpp::types::Float>
struct LineSearch : public ::nlpp::poly::CloneBase<LineSearch<Float>>,
					public ::nlpp::LineSearch<LineSearch<Float>>
{
	virtual Float lineSearch (::nlpp::wrap::LineSearch<::nlpp::wrap::poly::FunctionGradient<>, ::nlpp::Vec>) = 0;
};


template <typename Float>
struct LineSearch_ : public ::nlpp::poly::PolyClass<LineSearch<Float>>,
					 public ::nlpp::LineSearch<LineSearch_<Float>>
{
	NLPP_USING_POLY_CLASS(Base, ::nlpp::poly::PolyClass<LineSearch<Float>>);

	LineSearch_ () : Base(new StrongWolfe<Float>()) {}

	Float lineSearch (::nlpp::wrap::LineSearch<::nlpp::wrap::poly::FunctionGradient<>, ::nlpp::Vec> f)
	{
		return impl->lineSearch(f);
	}
};


} // namespace poly



//template <class Impl, bool Polymorphic, typename... Args>
//using LineSearchPick = std::conditional_t<Polymorphic, poly::LineSearch<Args...>, LineSearch<Impl>>;



} // namespace nlpp
