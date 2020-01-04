#pragma once

#include "../LineSearch.hpp"

#include "LineSearch/Interpolation/Interpolation.hpp"

#include "LineSearch/InitialStep/Constant.hpp"
#include "LineSearch/InitialStep/FirstOrder.hpp"


namespace nlpp
{

namespace impl
{

template <typename Float = types::Float, class InitialStep = ConstantStep<Float>>
struct StrongWolfe : public LineSearchBase<Float, InitialStep>
{
	using Base = LineSearchBase<Float, InitialStep>;
	using Base::f0;
	using Base::g0;
	using Base::initialStep;


	StrongWolfe (Float c1 = 1e-4, Float c2 = 0.9, const InitialStep& initialStep = InitialStep(), Float aMaxC = 100.0, 
				 Float rho = constants::phi, int maxIterBrack = 20, int maxIterInt = 1e2, Float tol = constants::eps_<Float>) :
				 c1(c1), c2(c2), Base(initialStep), aMaxC(aMaxC), rho(rho), maxIterBrack(maxIterBrack), maxIterInt(maxIterInt), tol(tol)
	{
		// assert(a0 > 0.0 && "a0 must be positive");
		assert(c1 > 0.0  && c2 > 0.0 && "c1 and c2 must be positive");
		assert(c1 < c2 && "c1 must be smaller than c2");
		// assert(a0 < aMaxC && "a0 must be smaller than aMaxC");
	}


	template <class Function>
	Float lineSearch (Function f)
	{
		std::tie(f0, g0) = f(0.0);

		Float a0 = initialStep(f0, g0);

		Float a = 0.0, fa = f0, ga = g0;
		Float b = a0, fb, gb;

		Float safeGuard = 0.0;

		int iter = 0;

		//Float aMax = aMaxC * std::max(xNorm, Float(N));
		Float aMax = 20.0;
		Float fMax = f.function(aMax);

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


			Float next = interpolate(b, aMax, fb, fMax, gb);

			a = b, fa = fb, ga = gb;

			if(next - std::sqrt(tol) <= b)
				b = b + rho * (b - a);
			
			else
				b = next;
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
			Float next = interpolate(l, u, fl, fu, gl, gu);

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

			if(u - l < 2 * tol)
				break;
		}

		return a;
	}


	// Float interpolate (Float a, Float fa, Float ga, Float b, Float fb, Float gb)
	// {
	// 	Float d1 = ga + gb - 3 * ((fa - fb) / (a - b));
	// 	Float d2 = sign(b - a) * std::sqrt(d1 * d1 - ga * gb);

	// 	Float next = b - (b - a) * ((gb + d2 - d1) / (gb - ga + 2 * d2));

	// 	if(std::isnan(next) || std::isinf(next))
	// 		return (a + b) / 2.0;
		
	// 	return next;
	// }



	Float c1;
	Float c2;
	Float aMaxC;
	Float rho;
	int maxIterBrack;
	int maxIterInt;
	Float tol;
};

} // namespace impl


template <typename Float = types::Float, class InitialStep = ConstantStep<Float>>
struct StrongWolfe : public impl::StrongWolfe<Float, InitialStep>,
					 public LineSearch<StrongWolfe<Float, InitialStep>>
{
	using Interface = LineSearch<StrongWolfe<Float, InitialStep>>;
	using Impl = impl::StrongWolfe<Float, InitialStep>;
	using Impl::Impl;

	void initialize ()
	{
		Impl::initialize();
	}

	template <class Function>
	auto lineSearch (Function f)
	{
		return Impl::lineSearch(f);
	}
};

namespace poly
{

template <class V = Vec, class InitialStep = ConstantStep<typename V::Scalar>>
struct StrongWolfe : public impl::StrongWolfe<::nlpp::impl::Scalar<V>, InitialStep>,
					 public LineSearchBase<V>
{
	using Float = ::nlpp::impl::Scalar<V>;

	using Interface = LineSearchBase<V>;
	using Impl = impl::StrongWolfe<Float, InitialStep>;
	using Impl::Impl;

	void initialize ()
	{
		Impl::initialize();
	}

	Float lineSearch (::nlpp::wrap::LineSearch<::nlpp::wrap::poly::FunctionGradient<V>, V> f)
	{
		return Impl::lineSearch(f);
	}

	virtual StrongWolfe* clone_impl () const {	return new StrongWolfe(*this);	}
};

} // namespace poly

} // namespace nlpp