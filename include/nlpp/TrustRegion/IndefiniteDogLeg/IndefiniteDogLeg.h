#pragma once

#include "../TrustRegion.h"

#include "../CauchyPoint/CauchyPoint.h"

#include "../DogLeg/DogLeg.h"

#include "../../Helpers/SpectraHelpers.h"


namespace nlpp
{

struct IndefiniteDogLeg : public TrustRegion<IndefiniteDogLeg>
{
	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, 
				   const Vec& x, double delta, double fx, const Vec& gx, const Mat& hx)
	{
		int N = x.rows();

		Vec v, u;

		Eigen::LLT<Mat> llt(hx);

		if(llt.info() == Eigen::NumericalIssue)
		{
			Eigen::EigenSolver<Mat> es(hx);

			double alpha = 1e20;
			int pos = 0;

			for(int i = 0; i < es.eigenvalues().rows(); ++i)
				if(es.eigenvalues().real()(i) < alpha)
					alpha = es.eigenvalues().real()(i), pos = i;
			
			Vec v1 = es.eigenvectors().real().col(pos);
			alpha = 2.0*abs(alpha);


			if(alpha < constants::eps)
			{
				//db("DDD");

				CauchyPoint cp;

				return cp.direction(function, gradient, hessian, x, delta, fx, gx, hx);
			}


			Vec dx = -(hx + alpha * Mat::Identity(N, N)).inverse() * gx;
			

			if(dx.norm() < delta)
			{
				double a = v1.dot(v1);
				double b = 2.0*dx.dot(v1);
				double c = dx.dot(dx) - delta*delta;

				double lx = (-b + std::sqrt(b*b - 4.0*a*c)) / (2.0*a), ux = (-b - std::sqrt(b*b - 4.0*a*c)) / (2.0*a);
				double fl = function(x + (dx + lx*v1)), fu = function(x + (dx + ux*v1));
				double e;

				e = fl < fu ? lx : ux;

				return dx + e * v1;
			}

			v = gx;
			u = -dx;

			//db("AAA");
		}

		else
		{//db("BBB");
			v = gx;
			u = hx.inverse() * gx;
		}


		Eigen::Vector2d g;
		g(0) = v.dot(gx);
		g(1) = u.dot(gx);
		
		Eigen::Matrix2d h;
		h(0, 0) = 2.0*v.transpose() * hx * v;
		h(1, 1) = 2.0*u.transpose() * hx * u;
		h(0, 1) = h(1, 0) = 2.0*v.transpose() * hx * u;
		

		Vec dir = -h.inverse() * g;

		dir = dir(0) * v + dir(1) * u;


		
		if(dir.norm() <= delta)
			return dir;

		

		//db("CCC");

		
		return findRoot(function, gradient, hessian, fx, x, gx, hx, delta);
	}

	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, double delta)
	{
		return this->operator()(function, gradient, hessian, x, delta, function(x), gradient(x), hessian(x));
	}


	template <class F, class G, class H>
	Vec findRoot (F f, G g, H h, double fx, const Vec& x, const Vec& gx, const Mat& hx, double delta)
	{
		std::complex<double> a = delta*delta;
		std::complex<double> b = 2.0 * a * hx.trace();
		std::complex<double> c = (a * std::pow(hx.trace(), 2.0) + 2.0 * a * hx.determinant() - gx.dot(gx));
		std::complex<double> d = (2.0 * a * hx.determinant() * hx.trace() - 2.0 * (gx.transpose() * hx.adjoint().transpose()).dot(gx));
		std::complex<double> e = (a * std::pow(hx.determinant(), 2.0) - (gx.transpose() * hx.adjoint().transpose()).dot(hx.adjoint() * gx));

		std::complex<double> p1 = 2.0*std::pow(c, 3.0) - 9.0*b*c*d + 27.0*a*std::pow(d, 2.0) + 27.0*std::pow(b, 2.0)*e - 72.0*a*c*e;
		std::complex<double> p2 = p1 + std::sqrt(-4.0*std::pow(std::pow(c, 2.0) -3.0*b*d + 12.0*a*e, 3.0) + std::pow(p1, 2.0));
		std::complex<double> p3 = ((std::pow(c, 2.0) - 3.0*b*d + 12.0*a*e) / (3.0*a*std::pow(p2/2.0, 1.0/3.0))) + ((std::pow(p2/2.0, 1.0/3.0)) / (3.0*a));
		std::complex<double> p4 = std::sqrt((std::pow(b, 2.0) / (4.0*std::pow(a, 2.0))) - ((2.0*c) / (3.0*a)) + p3);
		std::complex<double> p5 = (std::pow(b, 2.0) / (2.0*std::pow(a, 2.0))) - ((4.0*c) / (3.0*a)) - p3;
		std::complex<double> p6 = ((-std::pow(b, 3.0) / std::pow(a, 3.0)) + ((4.0*b*c) / std::pow(a, 2.0)) - ((8.0*d) / a)) / (4.0*p4);


		std::vector<std::complex<double>> roots(4);
		
		roots[0] = (-b / (4.0*a)) - (p4 / 2.0) - (std::sqrt(p5 - p6) / 2.0);
		roots[1] = (-b / (4.0*a)) - (p4 / 2.0) + (std::sqrt(p5 - p6) / 2.0);
		roots[2] = (-b / (4.0*a)) + (p4 / 2.0) - (std::sqrt(p5 + p6) / 2.0);
		roots[3] = (-b / (4.0*a)) + (p4 / 2.0) + (std::sqrt(p5 + p6) / 2.0);


		Vec dir = Vec::Constant(x.rows(), 0.0);
		double bestF = f(x + dir);

		for(int i = 0; i < roots.size(); ++i)
		{
			Vec aux = -(hx + roots[i].real() * Mat::Identity(hx.rows(), hx.rows())).inverse() * gx;

			//if(aux.norm() > delta)
			aux *= (delta / aux.norm());


			double faux = f(x + aux);

			if(faux < bestF)
				bestF = faux, dir = aux;
		}

		if(dir.norm() == 0.0)
		{
			//db("DDD");

			if(hx.llt().info() == Eigen::Success)
				return DogLeg().direction(f, g, h, x, delta, fx, gx, hx);
			
			return CauchyPoint().direction(f, g, h, x, delta, fx, gx, hx);
		}


		return dir;
	}
};

} // namespace nlpp
