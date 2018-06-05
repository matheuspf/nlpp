#pragma once

#include "../Helpers/Helpers.h"

#include "../Helpers/Optimizer.h"

#include "../LineSearch/Goldstein/Goldstein.h"


namespace nlpp
{

namespace params
{

template <class LineSearch = Goldstein>
struct GradientDescent : public GradientOptimizer<LineSearch>
{
	using GradientOptimizer<LineSearch>::GradientOptimizer;
};

} // namespace params


template <class LineSearch = Goldstein>
struct GradientDescent : public GradientOptimizer<GradientDescent<LineSearch>, params::GradientDescent<LineSearch>>
{
	CPPOPT_USING_PARAMS(Params, GradientOptimizer<GradientDescent<LineSearch>, params::GradientDescent<LineSearch>>);
	
	using Params::Params;
	

	template <class Function, typename Float, int Rows, int Cols>
	Vec optimize (Function f, Eigen::Matrix<Float, Rows, Cols> x)
	{
		auto [fx, gx] = f(x);

		Eigen::Matrix<Float, Rows, Cols> best = x;

		auto fBest = fx;

		Eigen::Matrix<Float, Rows, Cols> direction = -gx;


		for(int iter = 0; iter < maxIterations && direction.norm() > xTol; ++iter)
		{
			double alpha = lineSearch(f, x, direction);

			x = x + alpha * direction;

			fx = f(x, gx);

			if(fx < fBest)
				best = x;

			direction = -gx;
		}

		return best;
	}
};


} // namespace nlpp
