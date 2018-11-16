#pragma once

#include "../Helpers/Helpers.h"

#include "../Helpers/Optimizer.h"

#include "../LineSearch/StrongWolfe/StrongWolfe.h"

#include "Factorizations.h"


#define CPPOPT_USING_PARAMS_NEWTON(...) CPPOPT_USING_PARAMS(__VA_ARGS__);	\
										using Params::factorization;




namespace nlpp
{

namespace params
{

template <class Factorization = fact::SmallIdentity<>, class LineSearch = StrongWolfe,
		  class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
struct Newton : public GradientOptimizer<LineSearch, Stop, Output>
{
	CPPOPT_USING_PARAMS(Params, GradientOptimizer<LineSearch, Stop, Output>);
	using Params::Params;

	Factorization factorization;
};


} // namesapace params



template <class Factorization = fact::SmallIdentity<>, class LineSearch = StrongWolfe,
		  class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>>
struct Newton : public GradientOptimizer<Newton<Factorization, LineSearch, Stop, Output>,
								 params::Newton<Factorization, LineSearch, Stop, Output>>
{
	CPPOPT_USING_PARAMS_NEWTON(Params, GradientOptimizer<Newton<Factorization, LineSearch, Stop, Output>,
								 		 		 params::Newton<Factorization, LineSearch, Stop, Output>>);
	using Params::Params;


	template <class Function, class V, class Hessian>
	auto optimize (Function f, V x, Hessian hess)
	{
		auto[fx, gx] = f(x);

		stop.init(*this, x, fx, gx);
		output.init(*this, x, fx, gx);

		for(int iter = 0; iter < stop.maxIterations; ++iter)
		{
			auto dir = factorization(gx, hess(x));

			auto alpha = lineSearch(f, x, dir);

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

