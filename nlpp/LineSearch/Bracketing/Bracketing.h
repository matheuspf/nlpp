#pragma once

#include "../../helpers/helpers.hpp"


namespace nlpp
{


/** @brief Simple constant step bracketing of a single dimension function
*/
struct Bracketing
{
	/// The extrapolation constant. The default is the golden ratio
	Bracketing (double r = constants::phi) : r(r) {}


	/** @brief Given a function @c f and initial points @c a and @c b, return three points <tt>a, b and c</tt>,
	 * 		   such that @f$a < b < c$@f and @f$f(b) \leq f(a), \ f(b) \leq f(c)$@f. That is, there's a minimum
	 *  	   between @c a and @c b
	 * 
	 *  @param f The unidimensional function
	 *  @param a The lower initial point
	 *  @param b The upper initial point
	 * 
	 *  @note The search starts from @c a and always goes ahead. It never tries a number smaller than @c a.
	*/
	template <class Function>
	auto operator () (Function f, double a = 0.0, double b = 1.0) const
	{
		double fa = f(a);
		double fb = f(b);

		if(fa < fb)
		{
			std::swap(a, b);
			std::swap(fa, fb);
		}

		// First step is given by @f$ phi * (b - a)$@f from b
		double c = b + r * (b - a);
		double fc = f(c);

		while(fb > fc)
		{
			// Each new step, multiply by a constant greater than 1
			double d = b + r * (c - b);
			double fd = f(d);

			shift(a, b, c, d);
			shift(fa, fb, fc, fd);
		}

		if(c < a)
			std::swap(a, c);

		return std::make_tuple(a, b, c);
	}


	double r;	 ///< The extrapolation constant
};

} // namespace nlpp