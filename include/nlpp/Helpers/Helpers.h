/** @file
 *  @brief Some basic definitions and includes used by other files
*/
#pragma once

#include <type_traits>
#include <memory>
#include <deque>

#include "Include.h"
#include "Types.h"
#include "ForwardDeclarations.h"


namespace nlpp
{

namespace wrap
{

/** @name
 *  @brief Decides whether a given function has or has not a overloaded member functions taking the given parameters
*/
//@{
HAS_OVERLOADED_FUNC(operator(), HasOperator);

HAS_OVERLOADED_FUNC(function, HasFunction);

HAS_OVERLOADED_FUNC(gradient, HasGradient);
//@}

} // namespace wrap


namespace impl
{

/// A functor that does nothing
struct NullFunctor
{
	void operator () (...) {}
};


/** @name
 *  @brief Tells if @c T is an Eigen::EigenBase (a vector/matrix) or an scalar (a float or int)
*/
//@{
template <typename T>
struct IsMat
{
	template <class U>
	static constexpr bool impl (Eigen::EigenBase<U>*) { return true; }

	static constexpr bool impl (...) { return false; }


	enum { value = impl((std::decay_t<T>*)0) };
};

template <typename T>
struct IsScalar
{
	enum
	{
		value = (std::is_floating_point<std::decay_t<T>>::value || std::is_integral<std::decay_t<T>>::value)
	};
};

template <typename T>
constexpr bool isMat = IsMat<T>::value;

template <typename T>
constexpr bool isScalar = IsScalar<T>::value;
//@}



/** @name
 *  @brief Some aliases to avoid some typenames's/template's with Eigen types
*/
//@{
template <class V>
using Plain = typename V::PlainObject;

template <class V>
using PlainMatrix = typename V::PlainMatrix;

template <class V>
using PlainArray = typename V::PlainArray;

template <class V>
using Scalar = typename V::Scalar;

template <typename T, class V>
constexpr decltype(auto) cast (V&& v)
{
	return v.template cast<T>();
}
//@}


/** @name
 *  @brief Define function overloading calling precedence
*/
//@{
template <int I = 0, int Max = 10>
struct Precedence : Precedence<I+1, Max> {};

template <int I>
struct Precedence <I, I> {};
//@}


} // namespace impl



/// Definition of some constants
namespace constants
{

/// If boost is already included, use its defined constants
#ifdef BOOST_VERSION

#include <boost/math/constants/constants.hpp>

template <typename T = double>
constexpr T pi_ = boost::math::constants::pi<T>();

template <typename T = double>
constexpr T phi_ = boost::math::constants::phi<T>();

/// Else define our own
#else

template <typename T = types::Float>
constexpr T pi_ = T(3.14159265359);

template <typename T = types::Float>
constexpr T phi_ = T(1.61803398875);

#endif


template <typename T = types::Float>
constexpr T eps_ = T(1e-8);

constexpr double eps = eps_<types::Float>;
constexpr double pi  = pi_<types::Float>;
constexpr double phi = phi_<types::Float>;

} // namespace constants



/** @name
 *  @brief Useful for some line search operations
*/
//@{
template <typename T>
inline constexpr decltype(auto) shift (T&& x)
{
	return std::forward<T>(x);
}

template <typename T, typename U, typename... Args>
inline constexpr decltype(auto) shift (T&& x, U&& y, Args&&... args)
{
	x = std::forward<U>(y);

	return shift(std::forward<U>(y), std::forward<Args>(args)...);
}
//@}


/// Implement Matlab's sign function
template <typename T>
inline constexpr int sign (T t)
{
    return int(T{0} < t) - int(t < T{0});
}




} // namespace nlpp