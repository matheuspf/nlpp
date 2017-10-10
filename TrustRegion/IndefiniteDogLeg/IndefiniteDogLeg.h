#ifndef OPT_TR_INDEFINITE_DOGLEG_H
#define OPT_TR_INDEFINITE_DOGLEG_H

#include "../TrustRegion.h"

#include "../../SpectraHelpers.h"


struct IndefiniteDogLeg : public TrustRegion<IndefiniteDogLeg>
{
	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, 
				   const Vec& x, double delta, double fx, const Vec& gx, const Mat& hx)
	{
		int N = x.rows();

		Vec v, u;

		LLT<Mat> llt(hx);

		if(llt.info() == NumericalIssue)
		{
			EigenSolver<Mat> es(hx);

			double alpha = 1e20;
			int pos = 0;

			for(int i = 0; i < es.eigenvalues().rows(); ++i)
				if(es.eigenvalues().real()(i) < alpha)
					alpha = es.eigenvalues().real()(i), pos = i;
			
			Vec v1 = es.eigenvectors().real().col(pos);
			alpha = 2*abs(alpha);

			Vec dx = -(hx + alpha * Mat::Identity(N, N)).inverse() * gx;

			if(dx.norm() < delta)
			{
				double a = v1.dot(v1);
				double b = 2*dx.dot(v1);
				double c = dx.dot(dx) - delta*delta;

				double lx = (-b + sqrt(b*b - 4*a*c)) / (2*a), ux = (-b - sqrt(b*b - 4*a*c)) / (2*a);
				double fl = function(x + (dx + lx*v1)), fu = function(x + (dx + ux*v1));
				double e;

				e = fl < fu ? lx : ux;

				return dx + e * v1;
			}

			v = gx;
			u = -dx;

			//DB("AAA");
		}

		else
		{//db("BBB");
			v = gx;
			u = hx.inverse() * gx;
		}


		Vector2d g;
		g(0) = v.dot(gx);
		g(1) = u.dot(gx);
		
		Matrix2d h;
		h(0, 0) = 2*v.transpose() * hx * v;
		h(1, 1) = 2*u.transpose() * hx * u;
		h(0, 1) = h(1, 0) = 2*v.transpose() * hx * u;
		

		Vec dir = -h.inverse() * g;

		dir = dir(0) * v + dir(1) * u;

		//db("\n\n LEL    ", dir.norm(), "  ", delta, "   ", dir.norm() <= delta, "\n\n");
		
		if(dir.norm() <= delta)
			return dir;

		

		//DB("CCC");

		Matrix2d a;
		a(0, 0) = v.dot(v);
		a(1, 1) = u.dot(u);
		a(0, 1) = a(1, 0) = v.dot(u);
		
		dir = -(h + a).inverse() * g;

		dir = dir(0) * v + dir(1) * u;

		return dir * (delta / dir.norm());
	}

	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, double delta)
	{
		return this->operator()(function, gradient, hessian, x, delta, function(x), gradient(x), hessian(x));
	}
};




#endif // OPT_TR_INDEFINITE_DOGLEG_H