#pragma once

#include "../TrustRegion.h"

#include "../CauchyPoint/CauchyPoint.h"

#include "../DogLeg/DogLeg.h"

#include "../../Helpers/SpectraHelpers.h"


namespace nlpp
{

namespace impl
{

struct IndefiniteDogLeg
{
	template <class Function, class Hessian, class V, class M, typename Float>
	auto operator () (Function function, Hessian hessian, const V& x, const V& gx, const M& hx, Float delta)
	{
		int N = x.rows();

		V v, u;

		Eigen::LLT<Mat> llt(hx);

		if(llt.info() == Eigen::NumericalIssue)
		{
			Eigen::EigenSolver<Mat> es(hx);

			Float alpha = 1e20;
			int pos = 0;

			for(int i = 0; i < es.eigenvalues().rows(); ++i)
				if(es.eigenvalues().real()(i) < alpha)
					alpha = es.eigenvalues().real()(i), pos = i;
			
			V v1 = es.eigenvectors().real().col(pos);
			alpha = 2.0*abs(alpha);

			if(alpha < constants::eps)
				return CauchyPoint{}(function, hessian, x, gx, hx, delta);


			V dx = -(hx + alpha * Mat::Identity(N, N)).inverse() * gx;
			
			if(dx.norm() < delta)
			{
				Float a = v1.dot(v1);
				Float b = 2.0*dx.dot(v1);
				Float c = dx.dot(dx) - delta*delta;

				Float lx = (-b + std::sqrt(b*b - 4.0*a*c)) / (2.0*a);
				Float ux = (-b - std::sqrt(b*b - 4.0*a*c)) / (2.0*a);

				auto fl = function(x + (dx + lx*v1));
				auto fu = function(x + (dx + ux*v1));

				if(fl.first < fu.first)
					return std::tuple_cat(std::make_tuple((dx + lx * v1).eval()), fl);

				return std::tuple_cat(std::make_tuple((dx + ux * v1).eval()), fu);
			}

			v = gx;
			u = -dx;
		}

		else
		{
			v = gx;
			u = hx.inverse() * gx;
		}


		Eigen::Vector2d g;
		g(0) = v.dot(gx);
		g(1) = u.dot(gx);
		
		Eigen::Matrix2d h;
		h(0, 0) = 2.0*v.transpose() * hx * v;
		h(1, 1) = 2.0*u.transpose() * hx * u;
		h(0, 1) = h(1, 0) = 2.0*v.transpose() * hx * u;
		

		Vec dir = -h.inverse() * g;

		dir = dir(0) * v + dir(1) * u;
		
		if(dir.norm() <= delta)
			return std::tuple_cat(std::make_tuple(dir), function(x + dir));

		
		return findRoot(function, hessian, x, gx, hx, delta);
	}

	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, double delta)
	{
		return this->operator()(function, gradient, hessian, x, delta, function(x), gradient(x), hessian(x));
	}

	template <class Function, class Hessian, class V, class M, typename Float>
	auto findRoot (Function function, Hessian hessian, const V& x, const V& gx, const M& hx, Float delta)
	{
		std::complex<Float> a = delta*delta;
		std::complex<Float> b = 2.0 * a * hx.trace();
		std::complex<Float> c = (a * std::pow(hx.trace(), 2.0) + 2.0 * a * hx.determinant() - gx.dot(gx));
		std::complex<Float> d = (2.0 * a * hx.determinant() * hx.trace() - 2.0 * (gx.transpose() * hx.adjoint().transpose()).dot(gx));
		std::complex<Float> e = (a * std::pow(hx.determinant(), 2.0) - (gx.transpose() * hx.adjoint().transpose()).dot(hx.adjoint() * gx));

		std::complex<Float> p1 = 2.0*std::pow(c, 3.0) - 9.0*b*c*d + 27.0*a*std::pow(d, 2.0) + 27.0*std::pow(b, 2.0)*e - 72.0*a*c*e;
		std::complex<Float> p2 = p1 + std::sqrt(-4.0*std::pow(std::pow(c, 2.0) -3.0*b*d + 12.0*a*e, 3.0) + std::pow(p1, 2.0));
		std::complex<Float> p3 = ((std::pow(c, 2.0) - 3.0*b*d + 12.0*a*e) / (3.0*a*std::pow(p2/2.0, 1.0/3.0))) + ((std::pow(p2/2.0, 1.0/3.0)) / (3.0*a));
		std::complex<Float> p4 = std::sqrt((std::pow(b, 2.0) / (4.0*std::pow(a, 2.0))) - ((2.0*c) / (3.0*a)) + p3);
		std::complex<Float> p5 = (std::pow(b, 2.0) / (2.0*std::pow(a, 2.0))) - ((4.0*c) / (3.0*a)) - p3;
		std::complex<Float> p6 = ((-std::pow(b, 3.0) / std::pow(a, 3.0)) + ((4.0*b*c) / std::pow(a, 2.0)) - ((8.0*d) / a)) / (4.0*p4);


		std::vector<std::complex<Float>> roots(4);
		
		roots[0] = (-b / (4.0*a)) - (p4 / 2.0) - (std::sqrt(p5 - p6) / 2.0);
		roots[1] = (-b / (4.0*a)) - (p4 / 2.0) + (std::sqrt(p5 - p6) / 2.0);
		roots[2] = (-b / (4.0*a)) + (p4 / 2.0) - (std::sqrt(p5 + p6) / 2.0);
		roots[3] = (-b / (4.0*a)) + (p4 / 2.0) + (std::sqrt(p5 + p6) / 2.0);


		V best = Vec::Constant(x.rows(), 0.0);
		auto fBest = function(x + best);

		for(int i = 0; i < roots.size(); ++i)
		{
			V aux = -(hx + roots[i].real() * Mat::Identity(hx.rows(), hx.rows())).inverse() * gx;

			//if(aux.norm() > delta)
			aux *= (delta / aux.norm());


			auto fAux = function(x + aux);

			if(fAux.first < fBest.first)
				std::tie(best, fBest) = std::tie(aux, fAux);
		}

		if(best.norm() == 0.0)
		{
			if(hx.llt().info() == Eigen::Success)
				return DogLeg{}(function, hessian, x, gx, hx, delta);
			
			return CauchyPoint{}(function, hessian, x, gx, hx, delta);
		}

		return std::tuple_cat(std::make_tuple(best), fBest);
	}
};

} // namespace impl


template <class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>, typename Float = types::Float>
using IndefiniteDogLeg = TrustRegion<impl::IndefiniteDogLeg, Stop, Output, Float>;


namespace poly
{

namespace impl
{

template <class V = ::nlpp::Vec, class M = ::nlpp::Mat>
struct IndefiniteDogLeg : public LocalMinimizerBase<V, M>,
				public ::nlpp::impl::IndefiniteDogLeg
{
	using Interface = LocalMinimizerBase<V, M>;
	using Impl = ::nlpp::impl::IndefiniteDogLeg;
	using Float = ::nlpp::impl::Scalar<V>;

	virtual std::tuple<V, Float, V> operator () (::nlpp::wrap::poly::FunctionGradient<V> function, ::nlpp::wrap::poly::Hessian<V, M> hessian,
												 const V& x, const V& gx, const M& hx, Float delta)
	{
		return Impl::operator()(function, hessian, x, gx, hx, delta);
	}
};

} // namespace impl


template <class V = ::nlpp::Vec, class M = ::nlpp::Mat>
struct IndefiniteDogLeg : public TrustRegion<V, M>
{
	using Base = TrustRegion<V, M>;

	template <typename... Args>
	IndefiniteDogLeg (Args&&... args) : Base(std::forward<Args>(args)...)
	{
		Base::localOptimizer = std::make_unique<impl::IndefiniteDogLeg<V, M>>();
	}
};

} // namespace poly




} // namespace nlpp
