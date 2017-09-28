#ifndef OPT_TR_CAUCHY_POINT_H
#define OPT_TR_CAUCHY_POINT_H

#include "../TrustRegion.h"


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

		return min(1.0, pow(gxNorm, 3) / (delta * quad)) * dir;
	}


	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, double delta)
	{
		return this->operator()(function, gradient, hessian, x, delta, function(x), gradient(x), hessian(x));
	}
};




#endif // OPT_TR_CAUCHY_POINT_H