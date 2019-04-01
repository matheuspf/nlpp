#pragma once

#include "../TrustRegion.h"

#include "../../Helpers/SpectraHelpers.h"


namespace nlpp
{

namespace impl
{

struct IterativeTR
{
	template <class Function, class Hessian, class V, class M, typename Float>
	auto operator() (Function function, Hessian hessian, const V& x, const V& gx, const M& hx, Float delta)
	{
        int N = x.rows();
        V p, q;
        M In = Mat::Identity(N, N);

        Eigen::LLT<M> llt(hx);

        if(llt.info() == Eigen::Success)
        {
            p = -llt.solve(gx);

            if(p.norm() <= delta)
                return trReturn(function, x, p);
        }


        TopEigen<Spectra::SELECT_EIGENVALUE::SMALLEST_ALGE> topEigen(hx, 2);
        
        Float lambda1 = std::abs(topEigen.eigenvalues()(0));

        V v1 = topEigen.eigenvectors().col(0);


        if(std::abs(gx.dot(v1)) < std::sqrt(constants::eps_<Float>))
        {
            llt.compute(hx + lambda1 * In);

            p = -llt.solve(gx);
            p = p + v1 * ((delta - p.norm()) / v1.norm());

            return trReturn(function, x, p);
        }
        
        
        Float lambda = 2 * lambda1;
        int iters = maxIterations;

        while(iters--)
        {
            llt.compute(hx + lambda * In);

            p = -llt.solve(gx);

            if(std::abs(p.norm() - delta) <= terminationTol)
                return trReturn(function, x, p);


            q = static_cast<const M&>(llt.matrixL()). template triangularView<Eigen::Lower>().solve(p);
            
            lambda = lambda + (p.squaredNorm() / q.squaredNorm()) * ((p.norm() - delta) / delta);
        }

        return trReturn(function, x, p);
	}


    int maxIterations = 3;
    double terminationTol = 1e-2;
};

} // namespace impl


template <class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>, typename Float = types::Float>
using IterativeTR = TrustRegion<impl::IterativeTR, Stop, Output, Float>;


namespace poly
{

namespace impl
{

template <class V = ::nlpp::Vec, class M = ::nlpp::Mat>
struct IterativeTR : public LocalMinimizerBase<V, M>,
				     public ::nlpp::impl::IterativeTR
{
	using Interface = LocalMinimizerBase<V, M>;
	using Impl = ::nlpp::impl::IterativeTR;
	using Float = ::nlpp::impl::Scalar<V>;

	virtual std::tuple<V, Float, V> operator () (::nlpp::wrap::poly::FunctionGradient<V> function, ::nlpp::wrap::poly::Hessian<V, M> hessian,
												 const V& x, const V& gx, const M& hx, Float delta)
	{
		return Impl::operator()(function, hessian, x, gx, hx, delta);
	}
};

} // namespace impl


template <class V = ::nlpp::Vec, class M = ::nlpp::Mat>
struct IterativeTR : public TrustRegion<V, M>
{
	using Base = TrustRegion<V, M>;

	template <typename... Args>
	IterativeTR (Args&&... args) : Base(std::forward<Args>(args)...)
	{
		Base::localOptimizer = std::make_unique<impl::IterativeTR<V, M>>();
	}
};

} // namespace poly

} // namespace nlpp
