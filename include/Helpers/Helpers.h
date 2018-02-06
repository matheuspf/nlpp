#pragma once


#include <Eigen/Dense>

#include "../Handy/include/Handy.h"


namespace cppnlp
{

template <typename T>
using VecX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using MatX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

using Vec = VecX<double>;
using Mat = MatX<double>;

using Veci = VecX<int>;
using Mati = MatX<int>;



namespace constants
{

#ifdef BOOST_VERSION

#include <boost/math/constants/constants.hpp>

template <typename T = double>
constexpr T pi_ = boost::math::constants::pi<T>();

template <typename T = double>
constexpr T phi_ = boost::math::constants::phi<T>();

#else

template <typename T = double>
constexpr T pi_ = T(3.14159265359);

template <typename T = double>
constexpr T phi_ = T(1.61803398875);

#endif


template <typename T = double>
constexpr T eps_ = T(1e-8);

constexpr double eps = eps_<double>;
constexpr double pi  = pi_<double>;
constexpr double phi = phi_<double>;

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
    return (T{0} < t) - (t < T{0});
}


} // namespace cppnlp