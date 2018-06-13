#pragma once

#include "../LineSearch.h"


namespace nlpp
{

namespace impl
{

struct StrongWolfe
{
	StrongWolfe (double a0 = 1.0, double c1 = 1e-4, double c2 = 0.9, double aMaxC = 100.0, double rho = constants::phi, 
				 int maxIterBrack = 20, int maxIterInt = 1e2, double tol = constants::eps) :
				 a0(a0), c1(c1), c2(c2), aMaxC(aMaxC), rho(rho), 
				 maxIterBrack(maxIterBrack), maxIterInt(maxIterInt), tol(tol)
	{
		assert(a0 > 0.0 && "a0 must be positive");
		assert(c1 > 0.0  && c2 > 0.0 && "c1 and c2 must be positive");
		assert(c1 < c2 && "c1 must be smaller than c2");
		assert(a0 < aMaxC && "a0 must be smaller than aMaxC");
	}


	template <class Function>
	double lineSearch (Function f)
	{
		double f0, g0;
		
		std::tie(f0, g0) = f(0.0);

		double a = 0.0, fa = f0, ga = g0;
		double b = a0, fb, gb;

		double safeGuard = 0.0;

		int iter = 0;

		//double aMax = aMaxC * std::max(xNorm, double(N));
		double aMax = 100.0;

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


			double next = interpolate(a, fa, ga, b, fb, gb);

			a = b, fa = fb, ga = gb;

			b = next - tol <= b ? b + rho * (b - a) : next;
		}


		return safeGuard;
	}


	template <class Function>
	double zoom (Function f, double l, double fl, double gl, double u, double fu, double gu, double f0, double g0)
	{
		double a = l, fa, ga;

		int iter = 0;


		while(iter++ < maxIterInt)
		{
			double next = interpolate(l, fl, gl, u, fu, gu);

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


	double interpolate (double a, double fa, double ga, double b, double fb, double gb)
	{
		double d1 = ga + gb - 3 * ((fa - fb) / (a - b));
		double d2 = sign(b - a) * std::sqrt(d1 * d1 - ga * gb);

		return b - (b - a) * ((gb + d2 - d1) / (gb - ga + 2 * d2));
	}




	double a0;
	double c1, c2;
	double aMaxC;
	double rho;
	int maxIterBrack, maxIterInt;
	double tol;
};

} // namespace impl


struct StrongWolfe : public impl::StrongWolfe,
					 public LineSearch<StrongWolfe>
{
	using impl::StrongWolfe::StrongWolfe;

	template <class Function>
	double lineSearch (Function f)
	{
		return impl::StrongWolfe::lineSearch(f);
	}
};


namespace poly
{

template <class Function>
struct StrongWolfe : public impl::StrongWolfe,
					 public LineSearch<Function>
{
	using impl::StrongWolfe::StrongWolfe;

	double lineSearch (Function f)
	{
		return impl::StrongWolfe::lineSearch(f);
	}

	StrongWolfe* clone () const
	{
		return new StrongWolfe(*this);
	}
};

} // namespace poly


} // namespace nlpp
