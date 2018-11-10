#pragma once

#include "Include.h"


namespace nlpp
{

/// Default types
namespace types
{
	using Float = double;
	using Int = int;
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