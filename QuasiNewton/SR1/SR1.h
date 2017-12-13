#ifndef OPT_SR1_H
#define OPT_SR1_H

#include "../../Modelo.h"
#include "../../FiniteDifference.h"
#include "../../TrustRegion/TrustRegion.h"
#include "../../TrustRegion/IndefiniteDogLeg/IndefiniteDogLeg.h"

#include "../BFGS/BFGS.h"



template <class TR = IndefiniteDogLeg, class InitialHessian = BFGS_Diagonal>
struct SR1 : public TrustRegion<SR1<TR, InitialHessian>>
{
    using Base = TrustRegion<SR1<TR, InitialHessian>>;


    template <class Function, class Gradient>
	Vec operator () (Function f, Gradient g, const Vec& x)
	{
        B = initialHessian(f, g, x);
        B.diagonal().array() = 1.0 / B.diagonal().array();

        // H = B;
        // H.diagonal().array() = 1.0 / H.diagonal().array();

        return Base::operator()(f, g, [&](auto&&...){ return B; }, x);
	}

	template <class Function>
	Vec operator () (Function f, const Vec& x)
	{
		return this->operator()(f, gradientFD(f), x);
	}


    template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, 
				   double delta, double fx, const Vec& gx, Mat hx)
	{
        Vec s = trustRegion.direction(function, gradient, hessian, x, delta, fx, gx, hx);

        Vec y = gradient(x + s) - gradient(x);

        Vec d = y - B * s;

        double mag = r * s.norm() * d.norm();

        double sd = s.dot(d);

        if(abs(sd) >= mag)
        {
            B = B + (d * d.transpose()) / sd;
            
            // Vec dh = s - H * y;

            // H = H + (dh * dh.transpose()) / (dh.dot(y));
        }

        return s;
	}


	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, double delta)
	{
		return this->operator()(function, gradient, hessian, x, delta, function(x), gradient(x), hessian(x));
	}


    TR trustRegion;

    InitialHessian initialHessian;

    Mat B, H;

    double r = 1e-8;
};




#endif // OPT_SR1_H