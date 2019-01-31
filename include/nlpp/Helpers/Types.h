#pragma once

#include "Include.h"

#ifndef NLPP_FLOAT
	#define NLPP_FLOAT double
#endif

#ifndef NLPP_INT
	#define NLPP_INT int
#endif



namespace nlpp
{

/// Default types
namespace types
{
	using Float = NLPP_FLOAT;
	using Int = NLPP_INT;
}


/** @name
 *  @brief Default vector and matrix types
*/
//@{
template <typename T>
using VecX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using MatX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

using Vec = VecX<types::Float>;
using Mat = MatX<types::Float>;

using Veci = VecX<types::Int>;
using Mati = MatX<types::Int>;
//@}


} // namespace std