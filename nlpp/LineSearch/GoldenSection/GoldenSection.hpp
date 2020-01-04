/** @file
 *  @brief Golden section (Fibonacci) exact line search procedure
*/

#pragma once

#include "../../helpers/helpers.hpp"


namespace nlpp
{

/** @brief Golden section line search
 * 
 *  @details This is an exact 0th order line search algorithm, that is, does not need anything besides the
 * 			 function values.
 * 
 * 			 The idea is to split the area to search by creating two areas. An area is defined by three points, x_1, x_2 and
 * 			 x_3, such that x_1 < x_2 < x_3.
 * 
 * 			 Given an initial range [a, b], the starts by choosing two points x and y, such that @f$ x = a + q * (b - a) $@f
 * 			 and @f$ y = b - q * (b - a) $@f, where @f$ q = 1 - r @f$, and @f$ r = \frac{1}{\eps} $@f (@f$ \eps $@f is the 
 * 			 golden ratio).
 * 
 * 			 We then select the best of the two areas to search. If f(x) < f(y), (a, x, y) is the best area. Otherwise
 * 			 (x, y, b) is the best area.
 * 
 * 			 The algorithm proceeds while min{|b-y|, |x-a|} > 2 * tol, where tol is the square root of the machine precision,
 * 			 and while the maximum number of iterations is not exceeded.
 * 
 * @tparam Float Base floating point type
*/
template <typename Float = types::Float>
struct GoldenSection
{
	/// Default tolerance is square root of the machine precision, while maxIter has a high default value
	GoldenSection (Float tol = constants::eps, int maxIter = 10000) : tol(tol), maxIter(maxIter) {}


	/** @brief Executes the local search given a scalar function f and floats a, b and x, such that a < x < b. If only
	 * 		   a and b are given, first choose a point x.
	 * 
	 *  @param f Scalar function to be optimized
	 * 	@param a Lower bound of the search
	 *  @param b Upper bound of the search
	 *  @param x Middle point (optional)
	*/
	//@{
	template <class Function>
	Float operator () (Function f, Float a, Float b, Float x)
	{
		assert(x > a && x < b && "Wrong range for search");


		int iter = 0;
		Float y, fy, fx = f(x);


		if(std::abs(b - x) > std::abs(x - a))
			y = x + q * (b - x), fy = f(y);

		else
		{
			y = x - q * (x - a), fy = f(x);
			std::swap(x, y), std::swap(fx, fy);
		}


		//while(std::abs(b - a) > tol * (std::abs(x) + std::abs(y)) && ++iter < maxIter)
		while(std::min(std::abs(b - y), std::abs(x - a)) > 2 * tol && ++iter < maxIter)
		{
			if(fx < fy)
			{
				shift(b, y, x, a + r * (x - a));
				shift(fy, fx, f(x));
			}

			else
			{
				shift(a, x, y, b - r * (b - y));
				shift(fx, fy, f(y));
			}
		}

		return std::min(x, y, [&](Float x, Float y){ return f(x) < f(y); });
	}


	template <class Function>
	Float operator () (Function f, Float a, Float b)
	{
		assert(a < b && "Wrong range for search");


		Float x = a + q * (b - a);
		Float y = b - q * (b - a);

		Float fa = f(a), fb = f(b);
		Float fx = f(x), fy = f(y);


		if(fx < fy)
		{
			if(fx < fa && fx < fb)
				return operator()(f, a, b, x);

			return operator()(f, a, x, y);
		}

		else
		{
			if(fy < fa && fy < fb)
				return operator()(f, a, b, y);

			return operator()(f, x, b, y);
		}


		return std::nan;	// Quit CS
	}
	//@}



	Float tol;		///< Minimum tolerance on the values of x

	int maxIter;	///< Maximum number of function evaluations


	static constexpr Float r = 1.0 / constants::phi;		///< Inverse of the golden ratio
	static constexpr Float q = 1.0 - r;						///< @f$ 1 - \frac{1}{\eps} $@f
};

} // namespace nlpp
