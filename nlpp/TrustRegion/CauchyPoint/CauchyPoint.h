#pragma once

#include "../TrustRegion.h"


namespace nlpp
{

namespace impl
{

struct CauchyPoint
{
	template <class Function, class Hessian, class V, class M>
	auto operator() (Function function, Hessian hessian, const V& x, const V& gx, const M& hx, impl::Scalar<V> delta)
	{
		using Float = impl::Scalar<V>;

		Float gxNorm = gx.norm();

		V p = -(delta / gxNorm) * gx;

		Float quad = gx.transpose() * hx * gx;

		if(quad > constants::eps_<Float>)
			p = std::min(1.0, std::pow(gxNorm, 3) / (delta * quad)) * p;

		return std::tuple_cat(std::make_tuple(p), function(x + p));
	}
};

} // namespace impl

template <class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>, typename Float = types::Float>
using CauchyPoint = TrustRegion<impl::CauchyPoint, Stop, Output, Float>;


namespace poly
{

namespace impl
{

template <class V = ::nlpp::Vec, class M = ::nlpp::Mat>
struct CauchyPoint : public LocalMinimizerBase<V, M>,
						 public ::nlpp::impl::CauchyPoint
{
	using Interface = LocalMinimizerBase<V, M>;
	using Impl = ::nlpp::impl::CauchyPoint;
	using Float = ::nlpp::impl::Scalar<V>;

	virtual std::tuple<V, Float, V> operator () (::nlpp::wrap::poly::FunctionGradient<V> function, ::nlpp::wrap::poly::Hessian<V, M> hessian,
												 const V& x, const V& gx, const M& hx, Float delta)
	{
		return Impl::operator()(function, hessian, x, gx, hx, delta);
	}
};

} // namespace impl


template <class V = ::nlpp::Vec, class M = ::nlpp::Mat>
struct CauchyPoint : public TrustRegion<V, M>
{
	using Base = TrustRegion<V, M>;

	template <typename... Args>
	CauchyPoint (Args&&... args) : Base(std::forward<Args>(args)...)
	{
		Base::localOptimizer = std::make_unique<impl::CauchyPoint<V, M>>();
	}
};


} // namespace poly


} // namespace nlpp