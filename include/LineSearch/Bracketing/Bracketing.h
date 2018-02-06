#pragma once

#include "../../Helpers/Helpers.h"


namespace cppnlp
{

struct Bracketing
{
	Bracketing (double r = constants::phi) : r(r) {}


	template <class Function>
	auto operator () (Function f, double a = 0.0, double b = 1.0) const
	{
		double fa = f(a), fb = f(b);

		if(fa < fb)
			std::swap(a, b), std::swap(fa, fb);


		double c = b + r * (b - a), fc = f(c);

		while(fb > fc)
		{
			double d = b + r * (c - b), fd = f(d);

			shift(a, b, c, d);
			shift(fa, fb, fc, fd);
		}


		if(c < a)
			std::swap(a, c);

		return std::make_tuple(a, b, c);
	}


	double r;
};

} // namespace cppnlp