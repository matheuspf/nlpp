#pragma once

#include "../../Helpers/Helpers.h"

#include "../../Helpers/Optimizer.h"

#include "../../LineSearch/StrongWolfe/StrongWolfe.h"

#include "../BFGS/BFGS.h"


#define CPPOPT_USING_PARAMS_LBFGS(...) CPPOPT_USING_PARAMS(__VA_ARGS__);	\
                                       using Params::m;                     \
                                       using Params::initialHessian;


namespace nlpp
{

namespace impl
{

namespace params
{


template <class Params_, class InitialHessian = BFGS_Diagonal<>>
struct LBFGS : public Params_
{
	CPPOPT_USING_PARAMS(Params, Params_);
	using Params::Params;


	int m = 10;

    InitialHessian initialHessian;
};

} // namespace params


template <class Params_, class InitialHessian = BFGS_Diagonal<>>
struct LBFGS : public params::LBFGS<Params_, InitialHessian>
{
    CPPOPT_USING_PARAMS_BFGS(Params, params::LBFGS<Params_, InitialHessian>);
    using Params::Params;


	template <class Function, class V>
	V optimize (Function f, V x0)
	{
        using Float = impl::Scalar<V>;

        V gx(x0.rows(), x0.cols()), gx0(x0.rows(), x0.cols());
        Float fx0, fx;

        fx0 = f(x0, gx0);
        
        std::deque<V> vs;
        std::deque<V> vy;

        for(int iter = 0; iter < stop.maxIterations(); ++iter)
        {
            auto H = initialHessian(f, x0);

            V p = direction(f, x0, gx0, H, vs, vy);

            auto alpha = lineSearch(f, x0, p);

            V x = x0 + alpha * p;

            fx = f(x, gx);

            if(stop(*this, x, fx, gx))
                return x;

            auto s = x - x0;
            auto y = gx - gx0;

            vs.push_back(s);
            vy.push_back(y);

            if(iter > std::min(Params::m, (int)x0.size()))
            {
                vs.pop_front();
                vy.pop_front();
            }

            std::tie(x0, fx0, gx0) = std::tie(x, fx, gx);

            output(*this, x, fx, gx);
        }

        return x0;
	}


    template <class Function, class V, class U>
    V direction (Function f, const V& x, const V& gx, const Eigen::MatrixBase<U>& H,
                 const std::deque<V>& vs, const std::deque<V>& vy)
    {
        V alpha(vs.size());
        V rho(vs.size());
        V q = -gx;

        for(int i = vs.size() - 1; i >= 0; --i)
        {
            rho[i] = 1.0 / vy[i].dot(vs[i]);
            alpha[i] = rho[i] * vs[i].dot(q);

            q = q - alpha[i] * vy[i];
        }

        V r = H * q;

        for(int i = 0; i < vs.size(); ++i)
        {
            auto beta = rho[i] * vy[i].dot(r);

            r = r + (alpha[i] - beta) * vs[i];
        }

        return r;
    }
};

} // namespace impl



template <class InitialHessian = BFGS_Diagonal<>, class LineSearch = StrongWolfe<>,
          class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<0>>
struct LBFGS : public impl::LBFGS<params::GradientOptimizer<LineSearch, Stop, Output>, InitialHessian>,
			   public GradientOptimizer<LBFGS<InitialHessian, LineSearch, Stop, Output>>
{
    CPPOPT_USING_PARAMS(Impl, impl::LBFGS<params::GradientOptimizer<LineSearch, Stop, Output>, InitialHessian>);
    using Impl::Impl;

    template <class Function, class V>
    V optimize (Function f, V x)
    {
        return Impl::optimize(f, x);
    }
};


namespace poly
{

template <class InitialHessian = BFGS_Constant<>, class V = ::nlpp::Vec>
struct LBFGS : public ::nlpp::impl::LBFGS<::nlpp::poly::GradientOptimizer<V>, InitialHessian>
{
	CPPOPT_USING_PARAMS(Impl, ::nlpp::impl::LBFGS<::nlpp::poly::GradientOptimizer<V>, InitialHessian>);
	using Impl::Impl;

	virtual V optimize (::nlpp::wrap::poly::FunctionGradient<V> f, V x)
	{
		return Impl::optimize(f, x);
	}

	virtual LBFGS* clone_impl () const { return new LBFGS(*this); }
};

} // namespace poly


} // namespace nlpp
