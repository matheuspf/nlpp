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
        

        if(tryFactorize(gx, hx, p, delta))
            return std::tuple_cat(std::make_tuple(p), function(x + p));



        TopEigen<Spectra::SELECT_EIGENVALUE::SMALLEST_ALGE> topEigen(hx, 2);
        
        Float lambda = 2 * std::abs(topEigen.eigenvalues()(0));

        V v1 = topEigen.eigenvectors().col(0);


        if(std::abs(gx.dot(v1)) < 1e-4)
        {
            Eigen::LLT<M> llt(hx + std::abs(topEigen.eigenvalues()(0)) * In);

            p = -llt.solve(gx);

            p = p + v1 * ((delta - p.norm()) / v1.norm());

            return std::tuple_cat(std::make_tuple(p), function(x + p));
        }
        

        int maxIters = 3;

        while(maxIters--)
        {
            Eigen::LLT<M> llt(hx + lambda * In);

            // if(llt.info() == Eigen::NumericalIssue)
            // {
            //     maxIters = 3;
            //     lambda = 4 * std::abs(topEigen.eigenvalues()(0));
            //     continue;
            // }

            // db(abs(p.norm() - delta), "     ", (abs(p.norm() - delta) <= 1e-4));


            p = -llt.solve(gx);

            if(std::abs(p.norm() - delta) <= 1e-4)
                return std::tuple_cat(std::make_tuple(p), function(x + p));


            M L = llt.matrixL();

            q = L. template triangularView<Eigen::Lower>().solve(p);
            
            
            lambda = lambda + (p.squaredNorm() / q.squaredNorm()) * ((p.norm() - delta) / delta);
        }

        return std::tuple_cat(std::make_tuple(p), function(x + p));
	}


    template <class V, class M, typename Float>
    bool tryFactorize (const V& gx, const M& hx, V& p, Float delta)
    {
        Eigen::LLT<M> llt(hx);
        
        p = -llt.solve(gx);

        if(p.norm() <= delta)
            return true;

        return false;
    }
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
