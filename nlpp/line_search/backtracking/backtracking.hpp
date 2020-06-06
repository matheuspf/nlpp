#pragma once

#include "line_search/line_search.hpp"


namespace nlpp
{

/** @brief Backtracking line search
 * 
 *  @details Search for a point @c a satisfying the first of the Wolfe conditions given the function/gradient
 * 			 functor f
*/

namespace impl
{

template <template <class> class CRTP, typename Float = types::Float, class InitialStep = ConstantStep<Float>>
struct Backtracking : public CRTP<Backtracking<CRTP, Float, InitialStep>>
{
	/// Some reasonable default values
	Backtracking (const InitialStep& initialStep = InitialStep(1.0), Float c = 1e-4, Float rho = 0.5, Float aMin = constants::eps) :
				  initialStep(initialStep), c(c), rho(rho), aMin(aMin)
	{
		assert(c < 1.0 && "c must be smaller than 1.0");
		assert(rho < 1.0 && "rho must be smaller than 1.0");
	}


	/** @brief The line search procedure
	 *  @param f A function/gradient functor, projected on a single dimension
	*/
	template <class Function>
	Float lineSearch (Function f)
	{
		auto [f0, g0] = f(0.0);
		auto a = initialStep(*this, f0, g0);

		/// Check Wolfe's first condition @f$f(x + a * p) \leq f(x) + c_1 * a * p \intercall \nabla f(x)@f$
		while(a > aMin && f.function(a) > f0 + c * a * g0)
			a = rho * a;

		return a;
	}


	InitialStep initialStep;

	Float c;		///< Factor to control the linear Wolfe condition (@c c1)

	Float aMin;	///< Smallest step acceptable

	Float rho;		///< Factor to reduce @c a

};

} // namespace impl

template <typename Float = types::Float, class InitialStep = ConstantStep<Float>>
using Backtracking = impl::Backtracking<LineSearch, Float, InitialStep>;

} // namespace nlpp