#pragma once

#include "../Helpers/Helpers.h"

#include "../Helpers/Optimizer.h"

#include "../LineSearch/Goldstein/Goldstein.h"


namespace nlpp
{

namespace params
{

template <class LineSearch = Goldstein, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<0>>
struct GradientDescent : public GradientOptimizer<LineSearch, Stop, Output>
{
	CPPOPT_USING_PARAMS(Params, GradientOptimizer<LineSearch, Stop, Output>);
	using Params::Params;
};

} // namespace params


template <class LineSearch = Goldstein, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<0>>
struct GradientDescent : public GradientOptimizer<GradientDescent<LineSearch, Stop, Output>,
								params::GradientOptimizer<LineSearch, Stop, Output>>
{
	CPPOPT_USING_PARAMS(Params, GradientOptimizer<GradientDescent<LineSearch, Stop, Output>,
								params::GradientOptimizer<LineSearch, Stop, Output>>);
	using Params::Params;
	

	template <class Function, typename Float, int Rows, int Cols>
	Vec optimize (Function f, Eigen::Matrix<Float, Rows, Cols> x)
	{
		double fx;
		Eigen::Matrix<Float, Rows, Cols> gx, dir;
		
		std::tie(fx, gx) = f(x);

		stop.init(*this, x, fx, gx);
		output.init(*this, x, fx, gx);

		for(int iter = 0; iter < stop.maxIterations; ++iter)
		{
			dir = -gx;

			double alpha = lineSearch(f, x, dir);

			x = x + alpha * dir;

			fx = f(x, gx);

			if(stop(*this, x, fx, gx))
				break;

			output(*this, x, fx, gx);
		}

		output.finish(*this, x, fx, gx);

		return x;
	}
};


} // namespace nlpp
