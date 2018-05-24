#pragma once

#include "../Helpers/Helpers.h"

#include "../Helpers/Optimizer.h"

#include "../LineSearch/Goldstein/Goldstein.h"


namespace cppnlp
{

template <class LineSearch = Goldstein>
struct GradientDescent : public GradientOptimizer<GradientDescent<LineSearch>>
{
	using Base = GradientOptimizer<GradientDescent<LineSearch>>;
	using Base::Base;
	using Base::operator();

	GradientDescent (const LineSearch& lineSearch = LineSearch{}) : lineSearch(lineSearch) {}


	template <class Function, typename Float, int Rows, int Cols>
	Vec optimize (Function f, Eigen::Matrix<Float, Rows, Cols> x)
	{
		auto [fx, gx] = f(x);

		Eigen::Matrix<Float, Rows, Cols> best = x;

		auto fBest = fx;

		Eigen::Matrix<Float, Rows, Cols> direction = -gx;


		for(int iter = 0; iter < maxIterations && direction.norm() > xTol; ++iter)
		{
			double alpha = lineSearch([&](const auto& x){ return f.function(x); }, 
									  [&](const auto& x){ return f.gradient(x); }, x, direction);

			x = x + alpha * direction;

			fx = f(x, gx);

			if(fx < fBest)
				best = x;

			direction = -gx;
		}

		return best;
	}




	LineSearch lineSearch;


	int maxIterations = 1e3;

	double xTol = 1e-6;

	double fTol = 1e-7;
};

} // namespace cppnlp
