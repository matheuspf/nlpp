#pragma once

#include "projections_dec.hpp"


namespace nlpp
{

template <class V>
impl::Scalar<V> FR::operator () (const Eigen::MatrixBase<V>& fa, const Eigen::MatrixBase<V>& fb, const Eigen::MatrixBase<V>& dir = V()) const
{
    return fb.dot(fb) / fa.dot(fa);
}

template <class V>
impl::Scalar<V> PR::operator () (const Eigen::MatrixBase<V>& fa, const Eigen::MatrixBase<V>& fb, const Eigen::MatrixBase<V>& dir = V()) const
{
    return fb.dot(fb - fa) / fa.dot(fa);
}

template <class V>
impl::Scalar<V> PR_Abs::operator () (const Eigen::MatrixBase<V>& fa, const Eigen::MatrixBase<V>& fb, const Eigen::MatrixBase<V>& dir = V()) const
{
     return std::abs(PR::operator()(fa, fb));
}

template <class V>
impl::Scalar<V> PR_Plus::operator () (const Eigen::MatrixBase<V>& fa, const Eigen::MatrixBase<V>& fb, const Eigen::MatrixBase<V>& dir = V()) const
{
    return std::max(0.0, PR::operator()(fa, fb));
}

template <class V>
impl::Scalar<V> HS::operator () (const Eigen::MatrixBase<V>& fa, const Eigen::MatrixBase<V>& fb, const Eigen::MatrixBase<V>& dir = V()) const
{
     return fb.dot(fb - fa) / dir.dot(fb - fa);
}

template <class V>
impl::Scalar<V> DY::operator () (const Eigen::MatrixBase<V>& fa, const Eigen::MatrixBase<V>& fb, const Eigen::MatrixBase<V>& dir = V()) const
{
    return fb.dot(fb) / dir.dot(fb - fa);
}

template <class V>
impl::Scalar<V> HZ::operator () (const Eigen::MatrixBase<V>& fa, const Eigen::MatrixBase<V>& fb, const Eigen::MatrixBase<V>& dir = V()) const
{
    Vec y = fb - fa;

    auto yp = y.dot(dir);
	auto yy = y.dot(y);

	return (y - 2 * dir * (yy / yp)).dot(fb) / yp;
}

template <class V>
impl::Scalar<V> FR_PR::operator () (const Eigen::MatrixBase<V>& fa, const Eigen::MatrixBase<V>& fb, const Eigen::MatrixBase<V>& dir = V()) const
{
    auto fr = FR::operator()(fa, fb);
    auto pr = PR::operator()(fa, fb);

    if(pr < -fr)
        return -fr;

    if(std::abs(pr) <= fr)
        return pr;

    return fr;
}

} // namespace nlpp