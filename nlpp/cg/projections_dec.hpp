#pragma once

#include "../Helpers/Helpers.h"

/// Builds a functor, to avoid a lot of copy/paste
#define NLPP_CG_PROJECTION_DEC(...) \
struct __VA_ARGS__ \
{\
	template <class V>	\
	impl::Scalar<V> operator () (const Eigen::MatrixBase<V>& fa, const Eigen::MatrixBase<V>& fb, const Eigen::MatrixBase<V>& dir = V()) const;\
};

namespace nlpp
{

/** @name
 *  @brief The choice of the factor
*/
//@{
NLPP_CG_PROJECTION_DEC(FR)
NLPP_CG_PROJECTION_DEC(PR)
NLPP_CG_PROJECTION_DEC(PR_Abs : PR)
NLPP_CG_PROJECTION_DEC(PR_Plus : PR)
NLPP_CG_PROJECTION_DEC(HS)
NLPP_CG_PROJECTION_DEC(DY)
NLPP_CG_PROJECTION_DEC(HZ)
NLPP_CG_PROJECTION_DEC(FR_PR : FR, PR)
//@}

} // namespace nlpp

#undef NLPP_CG_PROJECTION_DEC