#pragma once

#include "../LineSearch.h"


namespace nlpp
{

/** @brief Backtracking line search
 * 
 *  @details Search for a point @c a satisfying the first of the Wolfe conditions given the function/gradient
 * 			 functor f
*/

namespace impl
{

template <typename Float = types::Float>
struct Backtracking
{
	/// Some reasonable default values
	Backtracking (Float a0 = 1.0, Float c = 1e-4, Float rho = 0.5, Float aMin = constants::eps) :
				  a0(a0), c(c), rho(rho), aMin(aMin)
	{
		assert(a0 > 1e-5 && "a0 must be larger than this");
		assert(c < 1.0 && "c must be smaller than 1.0");
		assert(rho < 1.0 && "rho must be smaller than 1.0");
		assert(aMin < a0 && "a0 must be greater than aMin");
	}


	/** @brief The line search procedure
	 *  @param f A function/gradient functor, projected on a single dimension
	*/
	template <class Function>
	Float lineSearch (Function f)
	{
		Float f0, g0, a = a0;

		std::tie(f0, g0) = f(0.0);

		/// Check Wolfe's first condition @f$f(x + a * p) \leq f(x) + c_1 * a * p \intercall \nabla f(x)@f$
		while(a > aMin && f.function(a) > f0 + c * a * g0)
			a = rho * a;

		return a;
	}


	Float a0;		///< Initial step

	Float c;		///< Factor to control the linear Wolfe condition (@c c1)

	Float aMin;	///< Smallest step acceptable

	Float rho;		///< Factor to reduce @c a
};

} // namespace impl


NLPP_LINE_SEARCH(Backtracking)


} // namespace nlpp