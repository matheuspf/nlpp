#ifndef OPT_STRONG_WOLFE_LS_H
#define OPT_STRONG_WOLFE_LS_H 

#include "../LineSearch.h"


struct StrongWolfe : public LineSearch<StrongWolfe>
{
	StrongWolfe (double a0 = 1.0, double c1 = 1e-4, double c2 = 0.9, double aMax = 10.0, 
				 double rho = goldenRatio, int maxIterBrack = 20, int maxIterInt = 1e2, double tol = EPS) :
				 a0(a0), c1(c1), c2(c2), aMax(aMax), rho(rho), 
				 maxIterBrack(maxIterBrack), maxIterInt(maxIterInt), tol(tol)
	{
		assert(a0 > 0.0 && "a0 must be positive");
		assert(c1 > 0.0  && c2 > 0.0 && "c1 and c2 must be positive");
		assert(c1 < c2 && "c1 must be smaller than c2");
		assert(a0 < aMax && "a0 must be smaller than aMax");
	}


	template <class Function, class Gradient>
	double lineSearch (Function f, Gradient g)
	{
		double f0 = f(0);
		double g0 = g(0);

		double a = 0.0, fa = f0, ga = g0;
		double b = a0, fb, gb;

		double safeGuard = 0.0;

		int iter = 0;


		while(iter++ < maxIterBrack && b + tol < aMax)
		{
			fb = f(b);
			gb = g(b);

			if(fb > f0 + b * c1 * g0 || (iter > 1 && fb > fa))
				return zoom(f, g, a, fa, ga, b, fb, gb, f0, g0);


			safeGuard = b;


			if(abs(gb) < c2 * abs(g0))
				return b;

			else if(gb > 0.0)
				return zoom(f, g, b, fb, gb, a, fa, ga, f0, g0);


			double next = interpolate(a, fa, ga, b, fb, gb);

			a = b, fa = fb, ga = gb;

			b = next - tol <= b ? b + rho * (b - a) : next;
		}


		return safeGuard;
	}


	template <class Function, class Gradient>
	double zoom (Function f, Gradient g, double l, double fl, double gl,
				 double u, double fu, double gu, double f0, double g0)
	{
		double a = l, fa, ga;

		int iter = 0;


		while(iter++ < maxIterInt)
		{
			double next = interpolate(l, fl, gl, u, fu, gu);

			if(a - tol <= l || a + tol >= u || abs(next - a) < tol)
				a = (u + l) / 2.0;

			else
				a = next;


			fa = f(a);
			ga = g(a);


			if(fa > f0 + a * c1 * g0 || fa > fl)
				u = a, fu = fa, gu = ga;

			else
			{
				if(abs(ga) < c2 * abs(g0))
					break;

				if(ga * (u - l) > 0.0)
					u = l, fu = fl, gu = gl;

				l = a, fl = fa, gl = ga;
			}

			if(u - l < 2 * EPS)
				break;
		}

		return a;
	}


	double interpolate (double a, double fa, double ga, double b, double fb, double gb)
	{
		double d1 = ga + gb - 3 * ((fa - fb) / (a - b));
		double d2 = SIGN(b - a) * sqrt(d1 * d1 - ga * gb);

		return b - (b - a) * ((gb + d2 - d1) / (gb - ga + 2 * d2));
	}




	double a0;
	double c1, c2;
	double aMax;
	double rho;
	int maxIterBrack, maxIterInt;
	double tol;
};




#endif // OPT_STRONG_WOLFE_LS_H