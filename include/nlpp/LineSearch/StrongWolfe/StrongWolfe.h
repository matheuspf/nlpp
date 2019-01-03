#pragma once

#include "../LineSearch.h"


namespace nlpp
{

namespace impl
{

template <typename Float>
struct StrongWolfe
{
	StrongWolfe (Float a0 = 1.0, Float c1 = 1e-4, Float c2 = 0.9, Float aMaxC = 100.0, Float rho = constants::phi, 
				 int maxIterBrack = 20, int maxIterInt = 1e2, Float tol = constants::eps) :
				 a0(a0), c1(c1), c2(c2), aMaxC(aMaxC), rho(rho), 
				 maxIterBrack(maxIterBrack), maxIterInt(maxIterInt), tol(tol)
	{
		assert(a0 > 0.0 && "a0 must be positive");
		assert(c1 > 0.0  && c2 > 0.0 && "c1 and c2 must be positive");
		assert(c1 < c2 && "c1 must be smaller than c2");
		assert(a0 < aMaxC && "a0 must be smaller than aMaxC");
	}


	template <class Function>
	Float lineSearch (Function f)
	{
		Float f0, g0;
		
		std::tie(f0, g0) = f(0.0);

		Float a = 0.0, fa = f0, ga = g0;
		Float b = a0, fb, gb;

		Float safeGuard = 0.0;

		int iter = 0;

		//Float aMax = aMaxC * std::max(xNorm, Float(N));
		Float aMax = 100.0;

		while(iter++ < maxIterBrack && b + tol < aMax)
		{
			std::tie(fb, gb) = f(b);

			if(fb > f0 + b * c1 * g0 || (iter > 1 && fb > fa))
				return zoom(f, a, fa, ga, b, fb, gb, f0, g0);


			safeGuard = b;


			if(std::abs(gb) < c2 * std::abs(g0))
				return b;

			else if(gb > 0.0)
				return zoom(f, b, fb, gb, a, fa, ga, f0, g0);


			Float next = interpolate(a, fa, ga, b, fb, gb);

			a = b, fa = fb, ga = gb;

			b = next - tol <= b ? b + rho * (b - a) : next;
		}


		return safeGuard;
	}


	template <class Function>
	Float zoom (Function f, Float l, Float fl, Float gl, Float u, Float fu, Float gu, Float f0, Float g0)
	{
		Float a = l, fa, ga;

		int iter = 0;


		while(iter++ < maxIterInt)
		{
			Float next = interpolate(l, fl, gl, u, fu, gu);

			if(a - tol <= l || a + tol >= u || std::abs(next - a) < tol)
				a = (u + l) / 2.0;

			else
				a = next;


			std::tie(fa, ga) = f(a);


			if(fa > f0 + a * c1 * g0 || fa > fl)
				u = a, fu = fa, gu = ga;

			else
			{
				if(std::abs(ga) < c2 * std::abs(g0))
					break;

				if(ga * (u - l) > 0.0)
					u = l, fu = fl, gu = gl;

				l = a, fl = fa, gl = ga;
			}

			if(u - l < 2 * constants::eps)
				break;
		}

		return a;
	}


	Float interpolate (Float a, Float fa, Float ga, Float b, Float fb, Float gb)
	{
		Float d1 = ga + gb - 3 * ((fa - fb) / (a - b));
		Float d2 = sign(b - a) * std::sqrt(d1 * d1 - ga * gb);

		return b - (b - a) * ((gb + d2 - d1) / (gb - ga + 2 * d2));
	}




	Float a0;
	Float c1;
	Float c2;
	Float aMaxC;
	Float rho;
	int maxIterBrack;
	int maxIterInt;
	Float tol;
};

} // namespace impl


NLPP_LINE_SEARCH_D(StrongWolfe)




} // namespace nlpp