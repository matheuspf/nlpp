#pragma once

#include "../LineSearch.h"


namespace cppnlp
{

struct GoldenSection
{
	GoldenSection (double tol = constants::eps, int maxIter = 1e4) : tol(tol), maxIter(maxIter) {}


	template <class Function>
	double operator () (Function f, double a, double b, double x)
	{
		assert(x > a && x < b && "Wrong range for search");


		int iter = 0;
		double y, fy, fx = f(x);


		if(std::abs(b - x) > std::abs(x - a))
			y = x + q * (b - x), fy = f(y);

		else
		{
			y = x - q * (x - a), fy = f(x);
			std::swap(x, y), std::swap(fx, fy);
		}


		//while(std::abs(b - a) > tol * (std::abs(x) + std::abs(y)) && ++iter < maxIter)
		while(std::abs(b - x) > 2 * tol && std::abs(x - a) > 2 * tol && ++iter < maxIter)
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

		return std::min(x, y, [&](double x, double y){ return f(x) < f(y); });
	}


	template <class Function>
	double operator () (Function f, double a, double b)
	{
		assert(a < b && "Wrong range for search");


		double x = a + q * (b - a);
		double y = b - q * (b - a);

		double fa = f(a), fb = f(b);
		double fx = f(x), fy = f(y);


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


		return x;		// Quit CS
	}



	double tol;

	int maxIter;


	static constexpr double r = 1.0 / constants::phi;
	static constexpr double q = 1.0 - r;
};

} // namespace cppnlp
