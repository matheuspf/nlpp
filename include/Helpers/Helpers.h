#pragma once


#include <Eigen/Dense>

#include "../Handy/include/Handy.h"


namespace cppnlp
{

namespace types
{
	using Float = double;
	using Int = int;
}





template <typename T>
using VecX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using MatX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

using Vec = VecX<types::Float>;
using Mat = MatX<types::Float>;

using Veci = VecX<types::Int>;
using Mati = MatX<types::Int>;



namespace impl
{

struct NullFunctor
{
	void operator () (...) {}
};


template <typename T>
struct IsMatImpl
{
	enum { value = false };
};

template <typename T>
struct IsMatImpl<Eigen::EigenBase<T>>
{
	enum { value = true };
};

template <typename T>
struct IsMat : public IsMatImpl<std::decay_t<T>> {};

template <typename T>
struct IsScalar
{
	enum{ value = !IsMat<T>::value };
};


template <typename T>
constexpr bool isMat = IsMat<T>::value;

template <typename T>
constexpr bool isScalar = isScalar<T>::value;


// template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
// struct IsMat<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
// {
// 	enum { value = true };
// };


} // namespace impl







namespace constants
{

#ifdef BOOST_VERSION

#include <boost/math/constants/constants.hpp>

template <typename T = double>
constexpr T pi_ = boost::math::constants::pi<T>();

template <typename T = double>
constexpr T phi_ = boost::math::constants::phi<T>();

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


template <typename T>
inline constexpr int sign (T t)
{
    return int(T{0} < t) - int(t < T{0});
}

} // namespace cppnlp