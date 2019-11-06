#pragma once

#include "../Helpers/Helpers.h"

#include "../Helpers/Optimizer.h"

#include "../LineSearch/StrongWolfe/StrongWolfe.h"

/// Builds a functor, to avoid a lot of copy/paste
#define BUILD_CG_STRUCT(Op, ...) \
\
struct __VA_ARGS__ \
{\
	template <class V>	\
	impl::Scalar<V> operator () (const V& fa, const V& fb, const V& dir = V()) const \
	{\
		Op; \
	}\
};



namespace nlpp
{

/** @name
 *  @brief The choice of the factor
*/
//@{
BUILD_CG_STRUCT(return fb.dot(fb) / fa.dot(fa), FR);

BUILD_CG_STRUCT(return fb.dot(fb - fa) / fa.dot(fa), PR);

BUILD_CG_STRUCT(return std::abs(PR::operator()(fa, fb)), PR_Abs : PR);

BUILD_CG_STRUCT(return std::max(0.0, PR::operator()(fa, fb)), PR_Plus : PR);

BUILD_CG_STRUCT(return fb.dot(fb - fa) / dir.dot(fb - fa), HS);

BUILD_CG_STRUCT(return fb.dot(fb) / dir.dot(fb - fa), DY);

BUILD_CG_STRUCT(Vec y = fb - fa;

				auto yp = y.dot(dir);
				auto yy = y.dot(y);

				return (y - 2 * dir * (yy / yp)).dot(fb) / yp;,

				HZ);

BUILD_CG_STRUCT(auto fr = FR::operator()(fa, fb);
				auto pr = PR::operator()(fa, fb);

				if(pr < -fr) return -fr;

				if(std::abs(pr) <= fr) return pr;

				return fr;,

				FR_PR : FR, PR);
//@}


namespace impl
{

template <class CGType, class Base_>
struct CG : public Base_
{
	NLPP_USING_LINESEARCH_OPTIMIZER(Base, Base_);
	using Base::Base;


} // namespace impl

} // namespace nlpp