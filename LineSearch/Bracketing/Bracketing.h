#ifndef OPT_BRACKETING_LS_H
#define OPT_BRACKETING_LS_H

#include "../../../Modelo.h"




struct Bracketing
{
	Bracketing (double r = goldenRatio) : r(r) {}


	template <class Function>
	auto operator () (Function f, double a = 0.0, double b = 1.0) const
	{
		double fa = f(a), fb = f(b);

		if(fa < fb)
			swap(a, b), swap(fa, fb);


		double c = b + r * (b - a), fc = f(c);

		while(fb > fc)
		{
			double d = b + r * (c - b), fd = f(d);

			shift(a, b, c, d);
			shift(fa, fb, fc, fd);
		}


		if(c < a)
			swap(a, c);

		return make_tuple(a, b, c);
	}


	double r;
};



#endif // OPT_BRACKETING_LS_H