#pragma once

#include "../TrustRegion.h"


namespace cppnlp
{

struct CauchyPoint : public TrustRegion<CauchyPoint>
{
	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, 
				   double delta, double fx, const Vec& gx, const Mat& hx)
	{
		double gxNorm = gx.norm();

		Vec dir = -(delta / gxNorm) * gx;

		double quad = gx.transpose() * hx * gx;


		if(quad <= 0.0)
			return dir;

		return std::min(1.0, std::pow(gxNorm, 3) / (delta * quad)) * dir;
	}


	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, double delta)
	{
		return this->operator()(function, gradient, hessian, x, delta, function(x), gradient(x), hessian(x));
	}
};

} // namespace cppnlp