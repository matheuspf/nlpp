#pragma once

#include "../Helpers/Helpers.h"

#include "../Helpers/Optimizer.h"

#include "../LineSearch/Goldstein/Goldstein.h"


namespace nlpp
{

namespace params
{

template <class LineSearch = Goldstein, class Output = out::GradientOptimizer>
struct GradientDescent : public GradientOptimizer<LineSearch, Output>
{
	using GradientOptimizer<LineSearch, Output>::GradientOptimizer;
};

} // namespace params


template <class LineSearch = Goldstein, class Output = out::GradientOptimizer>
struct GradientDescent : public GradientOptimizer<GradientDescent<LineSearch, Output>,
								params::GradientDescent<LineSearch, Output>>
{
	CPPOPT_USING_PARAMS(Params, GradientOptimizer<GradientDescent<LineSearch, Output>, params::GradientDescent<LineSearch, Output>>);
	
	using Params::Params;
	

	template <class Function, typename Float, int Rows, int Cols>
	Vec optimize (Function f, Eigen::Matrix<Float, Rows, Cols> x)
	{
		auto [fx, gx] = f(x);

		Eigen::Matrix<Float, Rows, Cols> best = x;

		auto fBest = fx;

		Eigen::Matrix<Float, Rows, Cols> direction = -gx;

		output.init(*this, fx);


		for(int iter = 0; iter < maxIterations && direction.norm() > xTol; ++iter)
		{
			double alpha = lineSearch(f, x, direction);

			x = x + alpha * direction;

			fx = f(x, gx);

			if(fx < fBest)
				best = x;

			direction = -gx;

			output(*this, fx);
		}

		output.finish(*this, fx);

		return best;
	}
};


} // namespace nlpp
