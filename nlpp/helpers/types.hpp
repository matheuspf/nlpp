#pragma once

#include "include.hpp"

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

} // namespace types


/** @name
 *  @brief Default vector and matrix types
*/
//@{
template <typename T, int Rows = Eigen::Dynamic>
using VecX = Eigen::Matrix<T, Rows, 1>;

template <typename T>
using MatX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

using Vec = VecX<types::Float>;
using Mat = MatX<types::Float>;

using Veci = VecX<types::Int>;
using Mati = MatX<types::Int>;
//@}

namespace impl
{

/** @name
 *  @brief Some aliases to avoid some typenames's/template's with Eigen types
*/
//@{
template <class V>
using Plain = typename std::decay_t<V>::PlainObject;

template <class V>
using PlainMatrix = typename std::decay_t<V>::PlainMatrix;

template <class V>
using PlainArray = typename std::decay_t<V>::PlainArray;

template <class V>
using Scalar = typename std::decay_t<V>::Scalar;

template <class V>
using Plain1D = Eigen::Matrix<Scalar<V>, std::decay_t<V>::RowsAtCompileTime, 1>;

template <class V>
using Ref = Eigen::Ref<Plain<V>>;
//@}

} // namespace impl

} // namespace std