#ifndef OPT_TR_INDEFINITE_DOGLEG_H
#define OPT_TR_INDEFINITE_DOGLEG_H

#include "../TrustRegion.h"


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
			alpha = -2*alpha;

			//db("\n", (hx + alpha * Mat::Identity(N, N)).inverse(), "\n");

			Vec dx = (hx + alpha * Mat::Identity(N, N)).inverse() * gx;

			if(dx.norm() < delta)
			{db("LEL   ", delta, "   ", (-dx + (v1 / v1.norm()) * (delta - dx.norm())).norm());
				return -dx + (v1 / v1.norm()) * (delta - dx.norm());
			}
asddddddddddddddddddddd
			v = gx;
			u = dx;
		}

		else
		{
			v = gx;
			u = hx.inverse() * gx;
		}

		v = gx;
		u = hx.inverse() * gx;

		Vector2d g;
		g(0) = v.dot(gx);
		g(1) = u.dot(gx);
		
		Matrix2d h;
		h(0, 0) = v.transpose() * hx * v;
		h(1, 1) = u.transpose() * hx * u;
		h(0, 1) = h(1, 0) = v.transpose() * hx * u;
		

		Vec dir = -h.inverse() * g;

		dir = dir(0) * v + dir(1) * u;

		//db("\n\n LEL    ", dir.norm(), "  ", delta, "   ", dir.norm() <= delta, "\n\n");

		if(dir.norm() <= delta)
			return dir;

		Matrix2d a;
		a(0, 0) = v.dot(v);
		a(1, 1) = u.dot(u);
		a(0, 1) = a(1, 0) = v.dot(u);
		
		dir = -(h + a).inverse() * g;

		dir = dir(0) * v + dir(1) * u;

		return dir;
	}

	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, double delta)
	{
		return this->operator()(function, gradient, hessian, x, delta, function(x), gradient(x), hessian(x));
	}
};




#endif // OPT_TR_INDEFINITE_DOGLEG_H