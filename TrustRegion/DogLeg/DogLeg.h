#ifndef OPT_TR_CAUCHY_POINT_H
#define OPT_TR_CAUCHY_POINT_H

#include "../TrustRegion.h"


struct DogLeg : public TrustRegion<DogLeg>
{
	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, 
				   double delta, double fx, const Vec& gx, Mat hx)
	{
		Vec pb = -hx.ldlt().solve(gx);

		if(pb.norm() <= delta)
			return pb;

			
		Vec pu = -(gx.dot(gx) / (gx.transpose() * hx * gx)) * gx;

		Vec diff = (pb - pu);


		double a = diff.squaredNorm();

		double b = 2 * (pu.dot(diff) - a);

		double c = a - pow(delta, 2) - 2 * pu.dot(diff) + pu.dot(pu);

		double d = sqrt(b*b - 4 * a * c);


		
		double tl = -(b + d) / (2 * a), tu =  -(b - d) / (2 * a), tau;

		Vec dl = pu + (tl - 1.0) * diff, du = pu + (tu - 1.0) * diff, dir;
		
		if(function(x + dl) < function(x + du))
			dir = dl;
		
		else
			dir = du;


		return dir;
	}


	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, double delta)
	{
		return this->operator()(function, gradient, hessian, x, delta, function(x), gradient(x), hessian(x));
	}
};




#endif // OPT_TR_CAUCHY_POINT_H