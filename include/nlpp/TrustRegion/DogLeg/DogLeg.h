#pragma once

#include "../TrustRegion.h"


namespace nlpp
{

namespace impl
{

struct DogLeg
{
	template <class Function, class Hessian, class V, class M, typename Float>
	auto operator() (Function function, Hessian hessia, const V& x, const V& gx, const M& hx, Float delta)
	{
		V pb = -hx.ldlt().solve(gx);

		if(pb.norm() <= delta)
			return std::tuple_cat(std::make_tuple(pb), function(x + pb));

		V pu = -(gx.dot(gx) / (gx.transpose() * hx * gx)) * gx;
		V diff = (pb - pu);

		Float a = diff.squaredNorm();
		Float b = 2 * (pu.dot(diff) - a);
		Float c = a - std::pow(delta, 2) - 2 * pu.dot(diff) + pu.dot(pu);
		Float d = std::sqrt(b*b - 4 * a * c);
		
		Float tl = -(b + d) / (2 * a);
		Float tu = -(b - d) / (2 * a);

		V dl = pu + (tl - 1.0) * diff;
		V du = pu + (tu - 1.0) * diff;

		auto lEval = function(x + dl);
		auto uEval = function(x + du);
		
		if(lEval.first < uEval.first)
			return std::tuple_cat(std::make_tuple(dl), lEval);
		
		return std::tuple_cat(std::make_tuple(du), uEval);
	}
};

} // namespace impl

template <class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>, typename Float = types::Float>
using DogLeg = TrustRegion<impl::DogLeg, Stop, Output, Float>;


namespace poly
{

namespace impl
{

template <class V = ::nlpp::Vec, class M = ::nlpp::Mat>
struct DogLeg : public LocalMinimizerBase<V, M>,
				public ::nlpp::impl::DogLeg
{
	using Interface = LocalMinimizerBase<V, M>;
	using Impl = ::nlpp::impl::DogLeg;
	using Float = ::nlpp::impl::Scalar<V>;

	virtual std::tuple<V, Float, V> operator () (::nlpp::wrap::poly::FunctionGradient<V> function, ::nlpp::wrap::poly::Hessian<V, M> hessian,
												 const V& x, const V& gx, const M& hx, Float delta)
	{
		return Impl::operator()(function, hessian, x, gx, hx, delta);
	}
};

} // namespace impl


template <class V = ::nlpp::Vec, class M = ::nlpp::Mat>
struct DogLeg : public TrustRegion<V, M>
{
	using Base = TrustRegion<V, M>;

	template <typename... Args>
	DogLeg (Args&&... args) : Base(std::forward<Args>(args)...)
	{
		Base::localOptimizer = std::make_unique<impl::DogLeg<V, M>>();
	}
};

} // namespace poly

} // namespace nlpp
