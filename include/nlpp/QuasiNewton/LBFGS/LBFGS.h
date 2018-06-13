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

namespace params
{

template <class InitialHessian = BFGS_Diagonal, class LineSearch = Goldstein, class Output = out::GradientOptimizer<0>>
struct LBFGS : public GradientOptimizer<LineSearch, Output>
{
	using Params = GradientOptimizer<LineSearch, Output>;
    using Params::Params;


	int m = 10;

    InitialHessian initialHessian;
};

} // namespace params


template <class Vec = nlpp::Vec, class InitialHessian = BFGS_Diagonal, 
          class LineSearch = StrongWolfe, class Output = out::GradientOptimizer<0>>
struct LBFGS : public GradientOptimizer<LBFGS<Vec, InitialHessian, LineSearch, Output>, 
                                        params::LBFGS<InitialHessian, LineSearch, Output>>
{
	CPPOPT_USING_PARAMS_LBFGS(Params, GradientOptimizer<LBFGS<Vec, InitialHessian, LineSearch, Output>, 
                                                        params::LBFGS<InitialHessian, LineSearch, Output>>);
	
	using Params::Params;
	

	template <class Function>
	Vec optimize (Function f, Vec x0)
	{
        auto [fx0, gx0] = f(x0);

        output.init(*this, fx0);

        for(int iter = 0; iter < maxIterations; ++iter)
        {
            auto H = initialHessian([&](const auto& x){ return f.function(x); },
                                    [&](const auto& x){ return f.gradient(x); }, x0);

            auto p = direction(f, x0, gx0, H);

            auto alpha = lineSearch(f, x0, p);

            Vec x = x0 + alpha * p;

            auto [fx, gx] = f(x);

            if(std::abs(fx - fx0) < fTol || gx.norm() < gTol || (x - x0).norm() < xTol)
                return x;

            Vec s = x - x0;
            Vec y = gx - gx0;

            vs.push_back(s);
            vy.push_back(y);

            if(iter > m)
            {
                vs.pop_front();
                vy.pop_front();
            }

            std::tie(x0, fx0, gx0) = std::tie(x, fx, gx);

            output(*this, fx);
        }

        output.finish(*this, fx0);

        return x0;
	}


    template <class Function, class Mat>
    Vec direction (Function f, const Vec& x, const Vec& gx, const Mat& H)
    {
        Vec alpha(vs.size());
        Vec rho(vs.size());
        Vec q = -gx;

        for(int i = vs.size() - 1; i >= 0; --i)
        {
            rho[i] = 1.0 / vy[i].dot(vs[i]);
            alpha[i] = rho[i] * vs[i].dot(q);

            q = q - alpha[i] * vy[i];
        }

        Vec r = H * q;

        for(int i = 0; i < vs.size(); ++i)
        {
            double beta = rho[i] * vy[i].dot(r);

            r = r + (alpha[i] - beta) * vs[i];
        }

        return r;
    }


    std::deque<Vec> vs;
    std::deque<Vec> vy;
};


} // namespace nlpp
